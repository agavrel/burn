use crate::{
    backend::{
        Dispatch, Metal, backend_extension,
        tensor::{FloatTensor, QuantizedTensor},
    },
    tensor::{DType, Shape, Tensor as BurnTensor},
};
use burn_cubecl::{CubeBackend, CubeRuntime, kernel::into_contiguous, tensor::CubeTensor};
use cubecl::{
    CubeCount, CubeDim, cube,
    prelude::*,
    quant::scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue},
};

/// Transformer operations that collapse common inference-only operator chains.
#[backend_extension(Metal)]
pub trait MetalTransformerBackend: crate::backend::Backend {
    /// RMS normalization over the last tensor dimension.
    fn fused_rms_norm(
        input: FloatTensor<Self>,
        gamma: FloatTensor<Self>,
        epsilon: f32,
    ) -> FloatTensor<Self>;

    /// Computes `silu(input) * gate` in one elementwise kernel.
    fn fused_silu_mul(input: FloatTensor<Self>, gate: FloatTensor<Self>) -> FloatTensor<Self>;

    /// Applies rotary position embeddings to a rank-four tensor.
    fn fused_rope(
        input: FloatTensor<Self>,
        frequencies: FloatTensor<Self>,
        sequence_dim: u32,
    ) -> FloatTensor<Self>;

    /// Matrix-vector multiplication for row-major `[input, output]` weights.
    fn fused_gemv(input: FloatTensor<Self>, weight: FloatTensor<Self>) -> FloatTensor<Self>;

    /// Matrix-vector multiplication with an output bias.
    fn fused_gemv_bias(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    /// Matrix-vector multiplication for packed block-Q8 `[input, output]` weights.
    fn fused_q8_gemv(input: FloatTensor<Self>, weight: QuantizedTensor<Self>) -> FloatTensor<Self>;

    /// Matrix-vector multiplication with packed block-Q8 weights and an output bias.
    fn fused_q8_gemv_bias(
        input: FloatTensor<Self>,
        weight: QuantizedTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

/// Applies a fused RMS normalization to a rank-three tensor.
pub fn rms_norm(input: BurnTensor<3>, gamma: BurnTensor<1>, epsilon: f64) -> BurnTensor<3> {
    BurnTensor::from_dispatch(Dispatch::fused_rms_norm(
        input.into_dispatch(),
        gamma.into_dispatch(),
        epsilon as f32,
    ))
}

/// Computes `silu(input) * gate` using one Metal dispatch.
pub fn silu_mul<const D: usize>(input: BurnTensor<D>, gate: BurnTensor<D>) -> BurnTensor<D> {
    BurnTensor::from_dispatch(Dispatch::fused_silu_mul(
        input.into_dispatch(),
        gate.into_dispatch(),
    ))
}

/// Applies RoPE to `[batch, heads, sequence, width]` (`sequence_dim = 2`) or
/// `[batch, sequence, heads, width]` (`sequence_dim = 1`) tensors.
pub fn apply_rope(
    input: BurnTensor<4>,
    frequencies: BurnTensor<3>,
    sequence_dim: usize,
) -> BurnTensor<4> {
    assert!(matches!(sequence_dim, 1 | 2));
    BurnTensor::from_dispatch(Dispatch::fused_rope(
        input.into_dispatch(),
        frequencies.into_dispatch(),
        sequence_dim as u32,
    ))
}

/// Applies a row-major Linear layer to rank-three one-token input.
pub fn linear(
    input: BurnTensor<3>,
    weight: BurnTensor<2>,
    bias: Option<BurnTensor<1>>,
) -> BurnTensor<3> {
    if let DType::QFloat(scheme) = weight.dtype() {
        if q8_gemv_block_size(scheme).is_none() {
            return crate::tensor::module::linear(input, weight, bias);
        }

        let output = match bias {
            Some(bias) => Dispatch::fused_q8_gemv_bias(
                input.into_dispatch(),
                weight.into_dispatch(),
                bias.into_dispatch(),
            ),
            None => Dispatch::fused_q8_gemv(input.into_dispatch(), weight.into_dispatch()),
        };
        return BurnTensor::from_dispatch(output);
    }

    let output = match bias {
        Some(bias) => Dispatch::fused_gemv_bias(
            input.into_dispatch(),
            weight.into_dispatch(),
            bias.into_dispatch(),
        ),
        None => Dispatch::fused_gemv(input.into_dispatch(), weight.into_dispatch()),
    };
    BurnTensor::from_dispatch(output)
}

fn q8_gemv_block_size(scheme: QuantScheme) -> Option<usize> {
    if !matches!(scheme.value, QuantValue::Q8F | QuantValue::Q8S)
        || scheme.param != QuantParam::F32
        || scheme.store != QuantStore::PackedU32(0)
        || scheme.mode != QuantMode::Symmetric
    {
        return None;
    }

    let QuantLevel::Block(block_size) = scheme.level else {
        return None;
    };
    let output = match block_size.as_slice() {
        [output] | [1, output] => *output as usize,
        _ => return None,
    };
    (output != 0 && output.is_multiple_of(4)).then_some(output)
}

#[cube(launch)]
fn rms_norm_kernel<F: Float>(
    input: &Tensor<F>,
    gamma: &Tensor<F>,
    output: &mut Tensor<F>,
    epsilon: f32,
    #[define(F)] _dtype: StorageType,
) {
    let row = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let width = input.shape(input.rank() - 1);
    let row_offset = row * width;
    let mut square_sum = F::new(0.0_f32);
    let mut column = lane;
    while column < width {
        let value = input[row_offset + column];
        square_sum += value * value;
        column += CUBE_DIM_X as usize;
    }
    let mean = plane_sum(square_sum) / F::cast_from(width as u32);
    let scale = (mean + F::cast_from(epsilon)).sqrt().recip();
    let mut column = lane;
    while column < width {
        output[row_offset + column] = input[row_offset + column] * scale * gamma[column];
        column += CUBE_DIM_X as usize;
    }
}

#[cube(launch)]
fn silu_mul_kernel<F: Float>(
    input: &Tensor<F>,
    gate: &Tensor<F>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS as usize;
    if index >= output.len() {
        terminate!();
    }
    let value = input[index];
    let sigmoid = F::new(1.0_f32) / (F::new(1.0_f32) + (F::new(0.0_f32) - value).exp());
    output[index] = value * sigmoid * gate[index];
}

#[cube(launch)]
fn rope_kernel<F: Float>(
    input: &Tensor<F>,
    frequencies: &Tensor<F>,
    output: &mut Tensor<F>,
    sequence_dim: u32,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS as usize;
    if index >= output.len() {
        terminate!();
    }

    let width = output.shape(3);
    let heads_or_sequence = output.shape(2);
    let sequence_or_heads = output.shape(1);
    let feature = index % width;
    let index_without_feature = index / width;
    let dim2 = index_without_feature % heads_or_sequence;
    let index_without_dim2 = index_without_feature / heads_or_sequence;
    let dim1 = index_without_dim2 % sequence_or_heads;
    let batch = index_without_dim2 / sequence_or_heads;
    let sequence = if sequence_dim == 1 { dim1 } else { dim2 };
    let pair_feature = feature - feature % 2;

    let input_base = batch * input.stride(0) + dim1 * input.stride(1) + dim2 * input.stride(2);
    let even = input[input_base + pair_feature * input.stride(3)];
    let odd = input[input_base + (pair_feature + 1) * input.stride(3)];
    let frequency_base = sequence * frequencies.stride(0) + feature * frequencies.stride(1);
    let cosine = frequencies[frequency_base];
    let sine = frequencies[frequency_base + frequencies.stride(2)];

    output[index] = if feature % 2 == 0 {
        even * cosine - odd * sine
    } else {
        odd * cosine + even * sine
    };
}

#[cube(launch)]
fn gemv_kernel<F: Float>(
    input: &Tensor<F>,
    weight: &Tensor<F>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let output_index = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let input_width = weight.shape(0);
    let output_width = weight.shape(1);
    let row = output_index / output_width;
    let column_out = output_index % output_width;
    let mut sum = F::new(0.0_f32);
    let mut column_in = lane;
    while column_in < input_width {
        sum += input[row * input_width + column_in] * weight[column_in * output_width + column_out];
        column_in += CUBE_DIM_X as usize;
    }
    let sum = plane_sum(sum);
    if lane == 0 {
        output[output_index] = sum;
    }
}

#[cube(launch)]
fn gemv_bias_kernel<F: Float>(
    input: &Tensor<F>,
    weight: &Tensor<F>,
    bias: &Tensor<F>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let output_index = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let input_width = weight.shape(0);
    let output_width = weight.shape(1);
    let row = output_index / output_width;
    let column_out = output_index % output_width;
    let mut sum = F::new(0.0_f32);
    let mut column_in = lane;
    while column_in < input_width {
        sum += input[row * input_width + column_in] * weight[column_in * output_width + column_out];
        column_in += CUBE_DIM_X as usize;
    }
    let sum = plane_sum(sum);
    if lane == 0 {
        output[output_index] = sum + bias[column_out];
    }
}

/// One SIMD group computes the four outputs stored in one packed `u32`. Its 32
/// lanes split the input dimension, then reduce four F32 accumulators. This
/// keeps enough groups active during one-token decoding while reading every
/// packed weight word only once.
#[cube(launch)]
fn q8_gemv_kernel<F: Float>(
    input: &Tensor<F>,
    weight: &Tensor<u32>,
    scales: &Tensor<f32>,
    output: &mut Tensor<F>,
    output_width: u32,
    block_size: u32,
    #[define(F)] _dtype: StorageType,
) {
    let lane = UNIT_POS_X as usize;
    let output_width = output_width as usize;
    let block_size = block_size as usize;
    let packed_output_width = output_width / 4;
    let row = CUBE_POS_X as usize / packed_output_width;
    let packed_column = CUBE_POS_X as usize % packed_output_width;
    let output_column = packed_column * 4;
    let input_width = weight.shape(0);
    let scale_column = output_column / block_size;
    let mut sum_0 = 0.0_f32;
    let mut sum_1 = 0.0_f32;
    let mut sum_2 = 0.0_f32;
    let mut sum_3 = 0.0_f32;
    let mut input_column = lane;

    while input_column < input_width {
        let packed = weight[input_column * weight.stride(0) + packed_column * weight.stride(1)];
        let raw_0 = packed & 255u32;
        let raw_1 = (packed >> 8u32) & 255u32;
        let raw_2 = (packed >> 16u32) & 255u32;
        let raw_3 = (packed >> 24u32) & 255u32;
        let quant_0 = raw_0 as i32 - (raw_0 >= 128u32) as i32 * 256;
        let quant_1 = raw_1 as i32 - (raw_1 >= 128u32) as i32 * 256;
        let quant_2 = raw_2 as i32 - (raw_2 >= 128u32) as i32 * 256;
        let quant_3 = raw_3 as i32 - (raw_3 >= 128u32) as i32 * 256;
        let scale = scales[input_column * scales.stride(0) + scale_column * scales.stride(1)];
        let value = f32::cast_from(input[row * input_width + input_column]) * scale;

        sum_0 += value * f32::cast_from(quant_0);
        sum_1 += value * f32::cast_from(quant_1);
        sum_2 += value * f32::cast_from(quant_2);
        sum_3 += value * f32::cast_from(quant_3);
        input_column += CUBE_DIM_X as usize;
    }

    let sum_0 = plane_sum(sum_0);
    let sum_1 = plane_sum(sum_1);
    let sum_2 = plane_sum(sum_2);
    let sum_3 = plane_sum(sum_3);
    if lane == 0 {
        let output_offset = row * output_width + output_column;
        output[output_offset] = F::cast_from(sum_0);
        output[output_offset + 1] = F::cast_from(sum_1);
        output[output_offset + 2] = F::cast_from(sum_2);
        output[output_offset + 3] = F::cast_from(sum_3);
    }
}

#[cube(launch)]
fn q8_gemv_bias_kernel<F: Float>(
    input: &Tensor<F>,
    weight: &Tensor<u32>,
    scales: &Tensor<f32>,
    bias: &Tensor<F>,
    output: &mut Tensor<F>,
    output_width: u32,
    block_size: u32,
    #[define(F)] _dtype: StorageType,
) {
    let lane = UNIT_POS_X as usize;
    let output_width = output_width as usize;
    let block_size = block_size as usize;
    let packed_output_width = output_width / 4;
    let row = CUBE_POS_X as usize / packed_output_width;
    let packed_column = CUBE_POS_X as usize % packed_output_width;
    let output_column = packed_column * 4;
    let input_width = weight.shape(0);
    let scale_column = output_column / block_size;
    let mut sum_0 = 0.0_f32;
    let mut sum_1 = 0.0_f32;
    let mut sum_2 = 0.0_f32;
    let mut sum_3 = 0.0_f32;
    let mut input_column = lane;

    while input_column < input_width {
        let packed = weight[input_column * weight.stride(0) + packed_column * weight.stride(1)];
        let raw_0 = packed & 255u32;
        let raw_1 = (packed >> 8u32) & 255u32;
        let raw_2 = (packed >> 16u32) & 255u32;
        let raw_3 = (packed >> 24u32) & 255u32;
        let quant_0 = raw_0 as i32 - (raw_0 >= 128u32) as i32 * 256;
        let quant_1 = raw_1 as i32 - (raw_1 >= 128u32) as i32 * 256;
        let quant_2 = raw_2 as i32 - (raw_2 >= 128u32) as i32 * 256;
        let quant_3 = raw_3 as i32 - (raw_3 >= 128u32) as i32 * 256;
        let scale = scales[input_column * scales.stride(0) + scale_column * scales.stride(1)];
        let value = f32::cast_from(input[row * input_width + input_column]) * scale;

        sum_0 += value * f32::cast_from(quant_0);
        sum_1 += value * f32::cast_from(quant_1);
        sum_2 += value * f32::cast_from(quant_2);
        sum_3 += value * f32::cast_from(quant_3);
        input_column += CUBE_DIM_X as usize;
    }

    let sum_0 = plane_sum(sum_0);
    let sum_1 = plane_sum(sum_1);
    let sum_2 = plane_sum(sum_2);
    let sum_3 = plane_sum(sum_3);
    if lane == 0 {
        let output_offset = row * output_width + output_column;
        output[output_offset] = F::cast_from(sum_0 + f32::cast_from(bias[output_column]));
        output[output_offset + 1] = F::cast_from(sum_1 + f32::cast_from(bias[output_column + 1]));
        output[output_offset + 2] = F::cast_from(sum_2 + f32::cast_from(bias[output_column + 2]));
        output[output_offset + 3] = F::cast_from(sum_3 + f32::cast_from(bias[output_column + 3]));
    }
}

fn empty_like<R: CubeRuntime>(tensor: &CubeTensor<R>) -> CubeTensor<R> {
    let shape = tensor.meta.shape().clone();
    let handle = tensor
        .client
        .empty(shape.num_elements() * tensor.dtype.size());
    CubeTensor::new_contiguous(
        tensor.client.clone(),
        tensor.device.clone(),
        shape,
        handle,
        tensor.dtype,
    )
}

fn elemwise_launch(rows: usize) -> (CubeCount, CubeDim) {
    let cube_dim = CubeDim::new_1d(256);
    let cubes = rows.div_ceil(cube_dim.x as usize) as u32;
    (CubeCount::Static(cubes, 1, 1), cube_dim)
}

fn gemv_output<R: CubeRuntime>(input: &CubeTensor<R>, weight: &CubeTensor<R>) -> CubeTensor<R> {
    assert_eq!(input.meta.num_dims(), 3);
    assert_eq!(weight.meta.num_dims(), 2);
    let input_width = input.meta.shape()[2];
    assert_eq!(weight.meta.shape()[0], input_width);
    let mut dims = input.meta.shape().to_vec();
    dims[2] = weight.meta.shape()[1];
    let shape = Shape::from(dims);
    let handle = input
        .client
        .empty(shape.num_elements() * input.dtype.size());
    CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        shape,
        handle,
        input.dtype,
    )
}

impl<R: CubeRuntime> MetalTransformerBackend for CubeBackend<R> {
    fn fused_rms_norm(
        input: FloatTensor<Self>,
        gamma: FloatTensor<Self>,
        epsilon: f32,
    ) -> FloatTensor<Self> {
        input.assert_is_on_same_device(&gamma);
        assert_eq!(input.dtype, gamma.dtype);
        let input = into_contiguous(input);
        let gamma = into_contiguous(gamma);
        let width = input.meta.shape()[input.meta.num_dims() - 1];
        assert_eq!(gamma.meta.num_elements(), width);
        let rows = input.meta.num_elements() / width;
        let output = empty_like(&input);
        rms_norm_kernel::launch(
            &input.client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new_1d(32),
            input.clone().into_tensor_arg(),
            gamma.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            epsilon,
            crate::backend::cubecl::dtype_to_storage_type(input.dtype),
        );
        output
    }

    fn fused_silu_mul(input: FloatTensor<Self>, gate: FloatTensor<Self>) -> FloatTensor<Self> {
        input.assert_is_on_same_device(&gate);
        assert_eq!(input.dtype, gate.dtype);
        assert_eq!(input.meta.shape(), gate.meta.shape());
        let input = into_contiguous(input);
        let gate = into_contiguous(gate);
        let output = empty_like(&input);
        let (cube_count, cube_dim) = elemwise_launch(output.meta.num_elements());
        silu_mul_kernel::launch(
            &input.client,
            cube_count,
            cube_dim,
            input.clone().into_tensor_arg(),
            gate.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            crate::backend::cubecl::dtype_to_storage_type(input.dtype),
        );
        output
    }

    fn fused_rope(
        input: FloatTensor<Self>,
        frequencies: FloatTensor<Self>,
        sequence_dim: u32,
    ) -> FloatTensor<Self> {
        input.assert_is_on_same_device(&frequencies);
        assert_eq!(input.dtype, frequencies.dtype);
        assert_eq!(input.meta.num_dims(), 4);
        assert_eq!(frequencies.meta.num_dims(), 3);
        let output = empty_like(&input);
        let (cube_count, cube_dim) = elemwise_launch(output.meta.num_elements());
        rope_kernel::launch(
            &input.client,
            cube_count,
            cube_dim,
            input.clone().into_tensor_arg(),
            frequencies.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            sequence_dim,
            crate::backend::cubecl::dtype_to_storage_type(input.dtype),
        );
        output
    }

    fn fused_gemv(input: FloatTensor<Self>, weight: FloatTensor<Self>) -> FloatTensor<Self> {
        input.assert_is_on_same_device(&weight);
        assert_eq!(input.dtype, weight.dtype);
        let input = into_contiguous(input);
        let weight = into_contiguous(weight);
        let output = gemv_output(&input, &weight);
        gemv_kernel::launch(
            &input.client,
            CubeCount::Static(output.meta.num_elements() as u32, 1, 1),
            CubeDim::new_1d(32),
            input.clone().into_tensor_arg(),
            weight.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            crate::backend::cubecl::dtype_to_storage_type(input.dtype),
        );
        output
    }

    fn fused_gemv_bias(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        input.assert_is_on_same_device(&weight);
        input.assert_is_on_same_device(&bias);
        assert_eq!(input.dtype, weight.dtype);
        assert_eq!(input.dtype, bias.dtype);
        let input = into_contiguous(input);
        let weight = into_contiguous(weight);
        let bias = into_contiguous(bias);
        let output = gemv_output(&input, &weight);
        assert_eq!(bias.meta.num_elements(), output.meta.shape()[2]);
        gemv_bias_kernel::launch(
            &input.client,
            CubeCount::Static(output.meta.num_elements() as u32, 1, 1),
            CubeDim::new_1d(32),
            input.clone().into_tensor_arg(),
            weight.into_tensor_arg(),
            bias.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            crate::backend::cubecl::dtype_to_storage_type(input.dtype),
        );
        output
    }

    fn fused_q8_gemv(input: FloatTensor<Self>, weight: QuantizedTensor<Self>) -> FloatTensor<Self> {
        input.assert_is_on_same_device(&weight);
        let scheme = match weight.dtype {
            DType::QFloat(scheme) => scheme,
            dtype => panic!("Expected a quantized Q8 weight, got {dtype:?}"),
        };
        let block_size = q8_gemv_block_size(scheme)
            .expect("Metal Q8 GEMV received an unsupported quantization scheme");
        let input = into_contiguous(input);
        let output = gemv_output(&input, &weight);
        let input_width = weight.meta.shape()[0];
        let output_width = weight.meta.shape()[1];
        assert!(output_width.is_multiple_of(4));

        let (values, scales) = weight
            .quantized_handles()
            .expect("Q8 weight must carry packed values and scales");
        assert_eq!(values.dtype, DType::U32);
        assert_eq!(values.meta.shape().dims(), [input_width, output_width / 4]);
        assert_eq!(scales.dtype, DType::F32);
        assert_eq!(
            scales.meta.shape().dims(),
            [input_width, output_width.div_ceil(block_size)]
        );

        let rows = input.meta.num_elements() / input_width;
        q8_gemv_kernel::launch(
            &input.client,
            CubeCount::Static((rows * output_width / 4) as u32, 1, 1),
            CubeDim::new_1d(32),
            input.clone().into_tensor_arg(),
            values.into_tensor_arg(),
            scales.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            output_width as u32,
            block_size as u32,
            crate::backend::cubecl::dtype_to_storage_type(input.dtype),
        );
        output
    }

    fn fused_q8_gemv_bias(
        input: FloatTensor<Self>,
        weight: QuantizedTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        input.assert_is_on_same_device(&weight);
        input.assert_is_on_same_device(&bias);
        assert_eq!(input.dtype, bias.dtype);
        let scheme = match weight.dtype {
            DType::QFloat(scheme) => scheme,
            dtype => panic!("Expected a quantized Q8 weight, got {dtype:?}"),
        };
        let block_size = q8_gemv_block_size(scheme)
            .expect("Metal Q8 GEMV received an unsupported quantization scheme");
        let input = into_contiguous(input);
        let bias = into_contiguous(bias);
        let output = gemv_output(&input, &weight);
        let input_width = weight.meta.shape()[0];
        let output_width = weight.meta.shape()[1];
        assert!(output_width.is_multiple_of(4));
        assert_eq!(bias.meta.num_elements(), output_width);

        let (values, scales) = weight
            .quantized_handles()
            .expect("Q8 weight must carry packed values and scales");
        assert_eq!(values.dtype, DType::U32);
        assert_eq!(values.meta.shape().dims(), [input_width, output_width / 4]);
        assert_eq!(scales.dtype, DType::F32);
        assert_eq!(
            scales.meta.shape().dims(),
            [input_width, output_width.div_ceil(block_size)]
        );

        let rows = input.meta.num_elements() / input_width;
        q8_gemv_bias_kernel::launch(
            &input.client,
            CubeCount::Static((rows * output_width / 4) as u32, 1, 1),
            CubeDim::new_1d(32),
            input.clone().into_tensor_arg(),
            values.into_tensor_arg(),
            scales.into_tensor_arg(),
            bias.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            output_width as u32,
            block_size as u32,
            crate::backend::cubecl::dtype_to_storage_type(input.dtype),
        );
        output
    }
}

use crate::{
    backend::{Dispatch, Metal, backend_extension, tensor::FloatTensor},
    tensor::{Shape, Tensor as BurnTensor},
};
use burn_cubecl::{CubeBackend, CubeRuntime, kernel::into_contiguous, tensor::CubeTensor};
use cubecl::{CubeCount, CubeDim, cube, prelude::*};

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
}

use crate::CubeRuntime;
use crate::{ops::empty_qtensor_optimized, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_elem_type;
use burn_backend::{
    TensorMetadata,
    quantization::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue},
};
use cubecl::{CubeCount, CubeDim, cube, prelude::*};

fn is_q4_k_block(scheme: &QuantScheme) -> Option<usize> {
    if scheme.value != QuantValue::Q4S
        || scheme.param != QuantParam::F32
        || scheme.store != QuantStore::PackedU32(0)
        || scheme.mode != QuantMode::Symmetric
    {
        return None;
    }
    let QuantLevel::Block(block) = scheme.level else {
        return None;
    };
    let [input, output] = block.as_dim::<2>();
    (input != 0 && output == 1).then_some(input as usize)
}

#[cube(launch)]
fn quantize_q4_k_block_kernel<F: Float>(
    input: &Tensor<F>,
    scales: &Tensor<F>,
    values_out: &mut Tensor<u32>,
    scales_out: &mut Tensor<f32>,
    output_width: u32,
    block_size: u32,
    #[define(F)] _dtype: StorageType,
) {
    let packed_index = ABSOLUTE_POS as usize;
    if packed_index >= values_out.len() {
        terminate!();
    }
    let output_width = output_width as usize;
    let packed_width = output_width / 8;
    let input_row = packed_index / packed_width;
    let packed_column = packed_index % packed_width;
    let output_column = packed_column * 8;
    let scale_row = input_row / block_size as usize;
    let values_stride_0 = values_out.stride(0);
    let values_stride_1 = values_out.stride(1);
    let scales_out_stride_0 = scales_out.stride(0);
    let scales_out_stride_1 = scales_out.stride(1);
    let mut packed = 0u32;

    #[unroll]
    for offset in 0..8usize {
        let column = output_column + offset;
        let scale_offset = scale_row * scales.stride(0) + column * scales.stride(1);
        let scale = scales[scale_offset];
        let input_offset = input_row * input.stride(0) + column * input.stride(1);
        let mut quant = (input[input_offset] / scale).round();
        if quant < F::new(-7.0_f32) {
            quant = F::new(-7.0_f32);
        }
        if quant > F::new(7.0_f32) {
            quant = F::new(7.0_f32);
        }
        let nibble = (i32::cast_from(quant) & 15) as u32;
        packed |= nibble << (offset * 4) as u32;

        if input_row.is_multiple_of(block_size as usize) {
            let output_scale_offset =
                scale_row * scales_out_stride_0 + column * scales_out_stride_1;
            scales_out[output_scale_offset] = f32::cast_from(scale);
        }
    }
    let output_value_offset = input_row * values_stride_0 + packed_column * values_stride_1;
    values_out[output_value_offset] = packed;
}

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
pub fn quantize<R>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
{
    let output = empty_qtensor_optimized(tensor.shape(), *scheme, &tensor.device);
    let (out_values, out_params) = output.clone().quantized_handles().unwrap();
    let dtype = tensor.dtype;

    if let Some(block_size) = is_q4_k_block(scheme) {
        let [input_width, output_width] = tensor.shape().dims();
        assert!(input_width.is_multiple_of(block_size));
        assert!(output_width.is_multiple_of(8));
        assert_eq!(scale.dtype, dtype);
        let cube_dim = CubeDim::new_1d(256);
        let work_items = input_width * output_width / 8;
        quantize_q4_k_block_kernel::launch(
            &output.client,
            CubeCount::Static(work_items.div_ceil(cube_dim.x as usize) as u32, 1, 1),
            cube_dim,
            tensor.into_tensor_arg(),
            scale.into_tensor_arg(),
            out_values.into_tensor_arg(),
            out_params.into_tensor_arg(),
            output_width as u32,
            block_size as u32,
            burn_backend::cubecl::dtype_to_storage_type(dtype),
        );
        return output;
    }

    cubek::quantization::quantize::launch_ref(
        &output.client,
        tensor.binding(),
        out_values.binding(),
        scale.binding(),
        out_params.binding(),
        scheme,
        dtype_to_elem_type(dtype),
    )
    .expect("Kernel to never fail");

    output
}

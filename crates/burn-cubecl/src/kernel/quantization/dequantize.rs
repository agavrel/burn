use crate::tensor::CubeTensor;
use crate::{CubeRuntime, ops::numeric::empty_device_dtype};
use alloc::{vec, vec::Vec};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{
    DType, TensorMetadata,
    quantization::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue},
};
use cubecl::{CubeCount, CubeDim, cube, prelude::*};

fn is_q4_k_block(scheme: QuantScheme) -> Option<usize> {
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
fn dequantize_q4_k_block_kernel<F: Float>(
    values: &Tensor<u32>,
    scales: &Tensor<f32>,
    output: &mut Tensor<F>,
    input_width: u32,
    output_width: u32,
    block_size: u32,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS as usize;
    if index >= output.len() {
        terminate!();
    }
    let output_width = output_width as usize;
    let input_width = input_width as usize;
    let flattened_input_row = index / output_width;
    let matrix = flattened_input_row / input_width;
    let input_row = flattened_input_row % input_width;
    let output_column = index % output_width;
    let packed_column = output_column / 8;
    let packed_width = output_width / 8;
    let packed = values[(matrix * input_width + input_row) * packed_width + packed_column];
    let shift = (output_column % 8) * 4;
    let raw = (packed >> shift as u32) & 15u32;
    let quant = raw as i32 - (raw >= 8u32) as i32 * 16;
    let scale_row = input_row / block_size as usize;
    let scale_rows = input_width / block_size as usize;
    let scale = scales[(matrix * scale_rows + scale_row) * output_width + output_column];
    output[index] = F::cast_from(scale * f32::cast_from(quant));
}

/// Convert the tensor back to a higher precision data type.
pub fn dequantize<R>(tensor: CubeTensor<R>, dtype: DType) -> CubeTensor<R>
where
    R: CubeRuntime,
{
    let scheme = match tensor.dtype {
        DType::QFloat(scheme) => scheme,
        _ => return tensor,
    };
    if let Some(block_size) = is_q4_k_block(scheme) {
        let shape = tensor.shape();
        let rank = shape.num_dims();
        assert!(rank >= 2);
        let input_width = shape[rank - 2];
        let output_width = shape[rank - 1];
        assert!(
            input_width.is_multiple_of(block_size),
            "Q4 K-block tensor shape {:?} is incompatible with block size {block_size} ({scheme:?})",
            tensor.shape()
        );
        assert!(output_width.is_multiple_of(8));
        let output = empty_device_dtype(
            tensor.client.clone(),
            tensor.device.clone(),
            tensor.shape(),
            dtype,
        );
        let (values, scales) = tensor.quantized_handles().unwrap();
        let cube_dim = CubeDim::new_1d(256);
        // Preserve every leading broadcast/batch matrix (for example the
        // rank-three weight produced by quantized matmul's unsqueeze).
        let work_items = shape.num_elements();
        dequantize_q4_k_block_kernel::launch(
            &output.client,
            CubeCount::Static(work_items.div_ceil(cube_dim.x as usize) as u32, 1, 1),
            cube_dim,
            values.into_tensor_arg(),
            scales.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            input_width as u32,
            output_width as u32,
            block_size as u32,
            dtype_to_storage_type(dtype),
        );
        return output;
    }
    let (tensor, inverse_axes) = match scheme.store {
        cubecl::quant::scheme::QuantStore::PackedU32(dim)
        | cubecl::quant::scheme::QuantStore::PackedNative(dim)
            if dim != 0 =>
        {
            let rank = tensor.rank();
            let packed_axis = rank - dim - 1;
            let mut axes = (0..rank)
                .filter(|axis| *axis != packed_axis)
                .collect::<Vec<_>>();
            axes.push(packed_axis);

            let mut inverse_axes = vec![0; rank];
            for (axis, source_axis) in axes.iter().enumerate() {
                inverse_axes[*source_axis] = axis;
            }

            let tensor = (packed_axis..rank - 1).fold(tensor, |tensor, axis| {
                crate::ops::swap_dims(tensor, axis, axis + 1)
            });

            (tensor, Some(inverse_axes))
        }
        _ => (tensor, None),
    };
    let scheme = tensor.scheme();

    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape(),
        dtype,
    );
    let (values, params) = tensor.quantized_handles().unwrap();

    cubek::quantization::dequantize::launch_ref(
        &output.client,
        values.binding(),
        output.clone().binding(),
        params.binding(),
        &scheme,
        dtype_to_storage_type(dtype),
    )
    .expect("Kernel to never fail");

    match inverse_axes {
        Some(axes) => crate::ops::permute(output, &axes),
        None => output,
    }
}

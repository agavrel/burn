use alloc::vec::Vec;
pub use burn_std::{QPARAM_ALIGN, params_shape};
use burn_std::{QuantLevel, QuantMode, QuantScheme, Shape};

use super::{Calibration, QuantizationParametersPrimitive};
use crate::{Backend, TensorMetadata, get_device_settings};

/// Rearrange a tensor so each logical quantization block is one contiguous row.
///
/// A plain `[num_blocks, block_elems]` reshape is only correct when the block
/// occupies a suffix of the tensor. For a block such as `[128, 1]` over a
/// `[input, output]` Linear weight it would mix output columns together. Split
/// every source dimension into `(block_count, block_width)`, move all counts
/// before all widths, then flatten the two groups independently.
fn blocks<B: Backend>(
    tensor: B::FloatTensorPrimitive,
    shape: &Shape,
    block_size: burn_std::BlockSize,
) -> B::FloatTensorPrimitive {
    let rank = shape.num_dims();
    let block_size = block_size.to_dim_vec(rank);
    let mut split_shape = Vec::with_capacity(rank * 2);

    for (&dim, &width) in shape.iter().zip(&block_size) {
        let width = width as usize;
        assert!(width != 0, "Quantization block dimensions must be non-zero");
        assert!(
            dim.is_multiple_of(width),
            "Tensor {shape:?} dimension {dim} must be evenly divisible by block dimension {width}"
        );
        split_shape.push(dim / width);
        split_shape.push(width);
    }

    let axes = (0..rank)
        .map(|dim| dim * 2)
        .chain((0..rank).map(|dim| dim * 2 + 1))
        .collect::<Vec<_>>();
    let num_blocks = shape
        .iter()
        .zip(&block_size)
        .map(|(&dim, &width)| dim / width as usize)
        .product();
    let block_elems = block_size.iter().map(|&width| width as usize).product();

    let tensor = B::float_reshape(tensor, Shape::from(split_shape));
    let tensor = B::float_permute(tensor, &axes);
    B::float_reshape(tensor, Shape::new([num_blocks, block_elems]))
}

/// Compute the quantization range mapping.
pub fn compute_range<B: Backend>(
    scheme: &QuantScheme,
    tensor: B::FloatTensorPrimitive,
    calibration: &Calibration,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    match calibration {
        Calibration::MinMax => match scheme.level {
            QuantLevel::Tensor => (B::float_min(tensor.clone()), B::float_max(tensor)),
            QuantLevel::Block(block_size) => {
                let shape = tensor.shape();
                let params_shape = params_shape(&shape, scheme.level);
                let blocks = blocks::<B>(tensor, &shape, block_size);
                let blocks_min =
                    B::float_reshape(B::float_min_dim(blocks.clone(), 1), params_shape.clone());
                let blocks_max = B::float_reshape(B::float_max_dim(blocks, 1), params_shape);
                (blocks_min, blocks_max)
            }
            QuantLevel::BlockTensor { .. } => {
                unimplemented!("two-level quantization is not supported yet")
            }
        },
        Calibration::AbsMean => {
            // gamma = mean(|W|) per tensor or block — symmetric range [-gamma, +gamma]
            let gamma = match scheme.level {
                QuantLevel::Tensor => B::float_mean(B::float_abs(tensor)),
                QuantLevel::Block(block_size) => {
                    let shape = tensor.shape();
                    let params_shape = params_shape(&shape, scheme.level);
                    let blocks = blocks::<B>(B::float_abs(tensor), &shape, block_size);
                    B::float_reshape(B::float_mean_dim(blocks, 1), params_shape)
                }
                QuantLevel::BlockTensor { .. } => {
                    unimplemented!("two-level quantization is not supported yet")
                }
            };
            let neg_gamma = B::float_neg(gamma.clone());
            (neg_gamma, gamma)
        }
    }
}

/// Compute the quantization parameters.
pub fn compute_q_params<B: Backend>(
    scheme: &QuantScheme,
    min: B::FloatTensorPrimitive,
    max: B::FloatTensorPrimitive,
) -> QuantizationParametersPrimitive<B> {
    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            ..
        } => {
            let bool_dtype = get_device_settings::<B>(&min.device()).bool_dtype;
            // Quantized range `[a, b]`
            let (a, b) = scheme.value.range();

            // Compute scale to convert an input value in range `[-alpha, alpha]`
            let min_abs = B::float_abs(min);
            let max_abs = B::float_abs(max);

            // `min_abs.max_pair(max_abs)`
            let mask = B::float_lower(min_abs.clone(), max_abs.clone(), bool_dtype);
            let values_range =
                B::float_mul_scalar(B::float_mask_where(min_abs, mask, max_abs), 2f32.into());

            QuantizationParametersPrimitive {
                scales: B::float_div_scalar(values_range, (b - a).into()),
            }
        }
        QuantScheme {
            level: QuantLevel::BlockTensor { .. },
            ..
        } => unimplemented!("two-level quantization is not supported yet"),
    }
}

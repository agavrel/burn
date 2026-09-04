use crate::{
    backend::{
        Dispatch, Metal, backend_extension,
        tensor::{FloatTensor, IntTensor, QuantizedTensor},
    },
    tensor::{DType, Int as BurnInt, Shape, Tensor as BurnTensor},
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

    /// Single-token grouped-query attention without materializing repeated KV heads.
    fn fused_grouped_query_attention_decode(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    /// Matrix-vector multiplication for row-major `[input, output]` weights.
    fn fused_gemv(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_width: u32,
    ) -> FloatTensor<Self>;

    /// Matrix-vector multiplication with an output bias.
    fn fused_gemv_bias(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    /// Matrix-vector multiplication for packed block-Q8 `[input, output]` weights.
    fn fused_q8_gemv(
        input: FloatTensor<Self>,
        weight: QuantizedTensor<Self>,
        output_width: u32,
    ) -> FloatTensor<Self>;

    /// Matrix-vector multiplication with packed block-Q8 weights and an output bias.
    fn fused_q8_gemv_bias(
        input: FloatTensor<Self>,
        weight: QuantizedTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    /// Matrix-vector multiplication for packed K-block Q4 `[input, output]` weights.
    fn fused_q4_gemv(
        input: FloatTensor<Self>,
        weight: QuantizedTensor<Self>,
        output_width: u32,
    ) -> FloatTensor<Self>;

    /// Matrix-vector multiplication with packed K-block Q4 weights and an output bias.
    fn fused_q4_gemv_bias(
        input: FloatTensor<Self>,
        weight: QuantizedTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    /// Gather rows from a packed hidden-axis block-Q4 embedding table.
    fn fused_q4_embedding(
        weight: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self>;

    /// Gather one row without allocating an indices tensor.
    fn fused_q4_embedding_row(weight: QuantizedTensor<Self>, row: u32) -> FloatTensor<Self>;

    /// Dot input rows with a semantic range plus one EOS row of a tied Q4 table.
    fn fused_q4_embedding_projection(
        input: FloatTensor<Self>,
        weight: QuantizedTensor<Self>,
        semantic_start: u32,
        semantic_rows: u32,
        eos_row: u32,
    ) -> FloatTensor<Self>;

    /// Samples one token with top-k/top-p filtering and returns
    /// `[token, next_random_cursor]` without synchronizing with the host.
    fn fused_sample_topk(
        logits: FloatTensor<Self>,
        random_scores: FloatTensor<Self>,
        random_cursor: IntTensor<Self>,
        temperature: f32,
        top_p: f32,
        top_k: u32,
    ) -> IntTensor<Self>;
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

/// Computes single-token grouped-query attention directly from compact KV heads.
pub fn grouped_query_attention_decode(
    query: BurnTensor<4>,
    key: BurnTensor<4>,
    value: BurnTensor<4>,
) -> BurnTensor<4> {
    BurnTensor::from_dispatch(Dispatch::fused_grouped_query_attention_decode(
        query.into_dispatch(),
        key.into_dispatch(),
        value.into_dispatch(),
    ))
}

/// Applies a row-major Linear layer to rank-three one-token input.
pub fn linear(
    input: BurnTensor<3>,
    weight: BurnTensor<2>,
    bias: Option<BurnTensor<1>>,
) -> BurnTensor<3> {
    let output_width = weight.dims()[1];
    if let DType::QFloat(scheme) = weight.dtype() {
        let output = if q8_gemv_block_size(scheme).is_some() {
            match bias {
                Some(bias) => Dispatch::fused_q8_gemv_bias(
                    input.into_dispatch(),
                    weight.into_dispatch(),
                    bias.into_dispatch(),
                ),
                None => Dispatch::fused_q8_gemv(
                    input.into_dispatch(),
                    weight.into_dispatch(),
                    output_width as u32,
                ),
            }
        } else if q4_gemv_block_size(scheme).is_some() {
            match bias {
                Some(bias) => Dispatch::fused_q4_gemv_bias(
                    input.into_dispatch(),
                    weight.into_dispatch(),
                    bias.into_dispatch(),
                ),
                None => Dispatch::fused_q4_gemv(
                    input.into_dispatch(),
                    weight.into_dispatch(),
                    output_width as u32,
                ),
            }
        } else {
            return crate::tensor::module::linear(input, weight, bias);
        };
        return BurnTensor::from_dispatch(output);
    }

    let output = match bias {
        Some(bias) => Dispatch::fused_gemv_bias(
            input.into_dispatch(),
            weight.into_dispatch(),
            bias.into_dispatch(),
        ),
        None => Dispatch::fused_gemv(
            input.into_dispatch(),
            weight.into_dispatch(),
            output_width as u32,
        ),
    };
    BurnTensor::from_dispatch(output)
}

/// Applies a bias-free Linear layer while producing only its leading output columns.
///
/// The weight keeps its full checkpoint-compatible shape. Specialized GEMV kernels
/// avoid reading or multiplying the omitted suffix.
pub fn linear_prefix(
    input: BurnTensor<3>,
    weight: BurnTensor<2>,
    output_width: usize,
) -> BurnTensor<3> {
    let full_output_width = weight.dims()[1];
    assert!(output_width > 0 && output_width <= full_output_width);
    if output_width == full_output_width {
        return linear(input, weight, None);
    }
    if let DType::QFloat(scheme) = weight.dtype() {
        let output = if q8_gemv_block_size(scheme).is_some() {
            Dispatch::fused_q8_gemv(
                input.into_dispatch(),
                weight.into_dispatch(),
                output_width as u32,
            )
        } else if q4_gemv_block_size(scheme).is_some() {
            Dispatch::fused_q4_gemv(
                input.into_dispatch(),
                weight.into_dispatch(),
                output_width as u32,
            )
        } else {
            let [batch, sequence, _] = input.dims();
            return crate::tensor::module::linear(input, weight, None).slice([
                0..batch,
                0..sequence,
                0..output_width,
            ]);
        };
        return BurnTensor::from_dispatch(output);
    }
    BurnTensor::from_dispatch(Dispatch::fused_gemv(
        input.into_dispatch(),
        weight.into_dispatch(),
        output_width as u32,
    ))
}

/// Gather rank-two token indices from an embedding table without expanding Q4 weights.
pub fn embedding(indices: BurnTensor<2, BurnInt>, weight: BurnTensor<2>) -> BurnTensor<3> {
    let DType::QFloat(scheme) = weight.dtype() else {
        return crate::tensor::module::embedding(weight, indices);
    };
    if !is_q4_embedding_scheme(scheme) {
        return crate::tensor::module::embedding(weight, indices);
    }
    BurnTensor::from_dispatch(Dispatch::fused_q4_embedding(
        weight.into_dispatch(),
        indices.into_dispatch(),
    ))
}

/// Gather one Q4 embedding row as `[1, 1, hidden]`.
pub fn embedding_row(weight: BurnTensor<2>, row: usize) -> BurnTensor<3> {
    let DType::QFloat(scheme) = weight.dtype() else {
        let width = weight.dims()[1];
        return weight.slice([row..row + 1, 0..width]).unsqueeze_dim::<3>(0);
    };
    assert!(
        is_q4_embedding_scheme(scheme),
        "Metal packed embedding lookup requires Q4S embedding blocks"
    );
    BurnTensor::from_dispatch(Dispatch::fused_q4_embedding_row(
        weight.into_dispatch(),
        row as u32,
    ))
}

/// Project hidden rows onto the tied semantic-token range and EOS row.
pub fn embedding_projection(
    input: BurnTensor<3>,
    weight: BurnTensor<2>,
    semantic_start: usize,
    semantic_rows: usize,
    eos_row: usize,
) -> BurnTensor<2> {
    let DType::QFloat(scheme) = weight.dtype() else {
        let width = weight.dims()[1];
        let semantic = weight
            .clone()
            .slice([semantic_start..semantic_start + semantic_rows, 0..width]);
        let eos = weight.slice([eos_row..eos_row + 1, 0..width]);
        let semantic_logits = input
            .clone()
            .matmul(semantic.swap_dims(0, 1).unsqueeze_dim::<3>(0))
            .select_dim(1, 0);
        let eos_logits = input
            .matmul(eos.swap_dims(0, 1).unsqueeze_dim::<3>(0))
            .select_dim(1, 0);
        return BurnTensor::cat([semantic_logits, eos_logits].into(), 1);
    };
    assert!(
        is_q4_embedding_scheme(scheme),
        "Metal packed embedding projection requires Q4S embedding blocks"
    );
    BurnTensor::from_dispatch(Dispatch::fused_q4_embedding_projection(
        input.into_dispatch(),
        weight.into_dispatch(),
        semantic_start as u32,
        semantic_rows as u32,
        eos_row as u32,
    ))
}

/// Samples one token while keeping the top-k, top-p, and random-race work on Metal.
///
/// `random_scores` contains precomputed Gumbel values in deterministic RNG order.
/// The returned tensor is `[token, next_random_cursor]`.
pub fn sample_topk(
    logits: BurnTensor<2>,
    random_scores: BurnTensor<1>,
    random_cursor: BurnTensor<1, BurnInt>,
    temperature: f64,
    top_p: f64,
    top_k: usize,
) -> BurnTensor<1, BurnInt> {
    assert!(temperature.is_finite() && temperature > 0.0);
    assert!(top_p.is_finite() && top_p > 0.0 && top_p <= 1.0);
    assert!(top_k > 0);
    let limit = top_k.min(logits.dims()[1]);
    BurnTensor::from_dispatch(Dispatch::fused_sample_topk(
        logits.into_dispatch(),
        random_scores.into_dispatch(),
        random_cursor.into_dispatch(),
        temperature as f32,
        top_p as f32,
        limit as u32,
    ))
}

fn is_q4_embedding_scheme(scheme: QuantScheme) -> bool {
    scheme.value == QuantValue::Q4S
        && scheme.param == QuantParam::F32
        && scheme.store == QuantStore::PackedU32(0)
        && scheme.mode == QuantMode::Symmetric
        && matches!(scheme.level, QuantLevel::Block(block) if block.as_dim::<2>() == [1, 32])
}

fn q4_gemv_block_size(scheme: QuantScheme) -> Option<usize> {
    if scheme.value != QuantValue::Q4S
        || scheme.param != QuantParam::F32
        || scheme.store != QuantStore::PackedU32(0)
        || scheme.mode != QuantMode::Symmetric
    {
        return None;
    }

    let QuantLevel::Block(block_size) = scheme.level else {
        return None;
    };
    let [input, output] = block_size.as_dim::<2>();
    (input != 0 && output == 1).then_some(input as usize)
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

/// Computes short-context attention for one Audio8 query head in one SIMD group.
#[cube(launch)]
fn grouped_query_attention_decode_short_kernel<F: Float, A: Float>(
    query: &Tensor<F>,
    key: &Tensor<F>,
    value: &Tensor<F>,
    output: &mut Tensor<F>,
    #[define(F, A)] _dtypes: [StorageType; 2],
) {
    let lane = UNIT_POS_X % 32;
    let query_head = CUBE_POS_X as usize % query.shape(1);
    let batch = CUBE_POS_X as usize / query.shape(1);
    let repeats = query.shape(1) / key.shape(1);
    let kv_head = query_head / repeats;
    let length = key.shape(2);
    let head_dim = query.shape(3);
    let feature_0 = lane as usize;
    let feature_1 = feature_0 + 32;
    let query_base = batch * query.stride(0) + query_head * query.stride(1);
    let key_base = batch * key.stride(0) + kv_head * key.stride(1);
    let value_base = batch * value.stride(0) + kv_head * value.stride(1);
    let query_0 = A::cast_from(query[query_base + feature_0 * query.stride(3)]);
    let query_1 = A::cast_from(query[query_base + feature_1 * query.stride(3)]);
    let scale = A::new(1.0_f32) / A::cast_from(head_dim as u32).sqrt();
    let mut local_max = A::new(-3.0e38_f32);
    let mut local_sum = A::new(0.0_f32);
    let mut local_output_0 = A::new(0.0_f32);
    let mut local_output_1 = A::new(0.0_f32);
    let mut position = 0usize;

    while position < length {
        let key_position = key_base + position * key.stride(2);
        let score = query_0 * A::cast_from(key[key_position + feature_0 * key.stride(3)])
            + query_1 * A::cast_from(key[key_position + feature_1 * key.stride(3)]);
        let score = plane_sum(score) * scale;
        if score > local_max {
            let previous_scale = (local_max - score).exp();
            local_output_0 *= previous_scale;
            local_output_1 *= previous_scale;
            local_sum *= previous_scale;
            local_max = score;
        }
        let position_scale = (score - local_max).exp();
        let value_position = value_base + position * value.stride(2);
        local_output_0 +=
            A::cast_from(value[value_position + feature_0 * value.stride(3)]) * position_scale;
        local_output_1 +=
            A::cast_from(value[value_position + feature_1 * value.stride(3)]) * position_scale;
        local_sum += position_scale;
        position += 1;
    }

    let output_stride = output.stride(3);
    let output_base = batch * output.stride(0) + query_head * output.stride(1);
    output[output_base + feature_0 * output_stride] = F::cast_from(local_output_0 / local_sum);
    output[output_base + feature_1 * output_stride] = F::cast_from(local_output_1 / local_sum);
}

/// Computes one of eight online-softmax shards for a long Audio8 query head.
#[cube(launch)]
fn grouped_query_attention_decode_partial_kernel<F: Float, A: Float>(
    query: &Tensor<F>,
    key: &Tensor<F>,
    value: &Tensor<F>,
    partials: &mut Tensor<A>,
    #[define(F, A)] _dtypes: [StorageType; 2],
) {
    let lane = UNIT_POS_X % 32;
    let partial = CUBE_POS_X as usize % 8;
    let query_index = CUBE_POS_X as usize / 8;
    let query_head = query_index % query.shape(1);
    let batch = query_index / query.shape(1);
    let repeats = query.shape(1) / key.shape(1);
    let kv_head = query_head / repeats;
    let length = key.shape(2);
    let head_dim = query.shape(3);
    let feature_0 = lane as usize;
    let feature_1 = feature_0 + 32;
    let query_base = batch * query.stride(0) + query_head * query.stride(1);
    let key_base = batch * key.stride(0) + kv_head * key.stride(1);
    let value_base = batch * value.stride(0) + kv_head * value.stride(1);
    let query_0 = A::cast_from(query[query_base + feature_0 * query.stride(3)]);
    let query_1 = A::cast_from(query[query_base + feature_1 * query.stride(3)]);
    let scale = A::new(1.0_f32) / A::cast_from(head_dim as u32).sqrt();
    let mut local_max = A::new(-3.0e38_f32);
    let mut local_sum = A::new(0.0_f32);
    let mut local_output_0 = A::new(0.0_f32);
    let mut local_output_1 = A::new(0.0_f32);
    let mut position = partial;

    while position < length {
        let key_position = key_base + position * key.stride(2);
        let score = query_0 * A::cast_from(key[key_position + feature_0 * key.stride(3)])
            + query_1 * A::cast_from(key[key_position + feature_1 * key.stride(3)]);
        let score = plane_sum(score) * scale;
        if score > local_max {
            let previous_scale = (local_max - score).exp();
            local_output_0 *= previous_scale;
            local_output_1 *= previous_scale;
            local_sum *= previous_scale;
            local_max = score;
        }
        let position_scale = (score - local_max).exp();
        let value_position = value_base + position * value.stride(2);
        local_output_0 +=
            A::cast_from(value[value_position + feature_0 * value.stride(3)]) * position_scale;
        local_output_1 +=
            A::cast_from(value[value_position + feature_1 * value.stride(3)]) * position_scale;
        local_sum += position_scale;
        position += 8;
    }

    let partial_base =
        batch * partials.stride(0) + query_head * partials.stride(1) + partial * partials.stride(2);
    let partial_stride = partials.stride(3);
    if lane == 0 {
        partials[partial_base] = local_max;
        partials[partial_base + partial_stride] = local_sum;
    }
    partials[partial_base + (feature_0 + 2) * partial_stride] = local_output_0;
    partials[partial_base + (feature_1 + 2) * partial_stride] = local_output_1;
}

/// Merges the eight online-softmax shards after the partial dispatch.
#[cube(launch)]
fn grouped_query_attention_decode_merge_kernel<F: Float, A: Float>(
    partials: &Tensor<A>,
    output: &mut Tensor<F>,
    #[define(F, A)] _dtypes: [StorageType; 2],
) {
    let lane = UNIT_POS_X % 32;
    let query_head = CUBE_POS_X as usize % output.shape(1);
    let batch = CUBE_POS_X as usize / output.shape(1);
    let feature_0 = lane as usize;
    let feature_1 = feature_0 + 32;
    let partial_base = batch * partials.stride(0) + query_head * partials.stride(1);
    let partial_stride = partials.stride(2);
    let value_stride = partials.stride(3);
    let mut maximum = partials[partial_base];
    let mut partial = 1usize;
    while partial < 8 {
        let candidate = partials[partial_base + partial * partial_stride];
        if candidate > maximum {
            maximum = candidate;
        }
        partial += 1;
    }

    let mut denominator = A::new(0.0_f32);
    let mut numerator_0 = A::new(0.0_f32);
    let mut numerator_1 = A::new(0.0_f32);
    partial = 0;
    while partial < 8 {
        let base = partial_base + partial * partial_stride;
        let merge_scale = (partials[base] - maximum).exp();
        denominator += partials[base + value_stride] * merge_scale;
        numerator_0 += partials[base + (feature_0 + 2) * value_stride] * merge_scale;
        numerator_1 += partials[base + (feature_1 + 2) * value_stride] * merge_scale;
        partial += 1;
    }

    let output_stride = output.stride(3);
    let output_base = batch * output.stride(0) + query_head * output.stride(1);
    output[output_base + feature_0 * output_stride] = F::cast_from(numerator_0 / denominator);
    output[output_base + feature_1 * output_stride] = F::cast_from(numerator_1 / denominator);
}

#[cube(launch)]
fn gemv_kernel<F: Float>(
    input: &Tensor<F>,
    weight: &Tensor<F>,
    output: &mut Tensor<F>,
    output_width: u32,
    #[define(F)] _dtype: StorageType,
) {
    let output_index = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let input_width = weight.shape(0);
    let output_width = output_width as usize;
    let row = output_index / output_width;
    let column_out = output_index % output_width;
    let mut sum = F::new(0.0_f32);
    let mut column_in = lane;
    while column_in < input_width {
        sum += input[row * input_width + column_in]
            * weight[column_in * weight.stride(0) + column_out * weight.stride(1)];
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

/// One SIMD group computes the eight output columns stored in one Q4 `u32`.
/// Scales vary along K, so every accumulator uses its output column's scale
/// for the current input block.
#[cube(launch)]
fn q4_gemv_kernel<F: Float>(
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
    let packed_output_width = output_width / 8;
    let row = CUBE_POS_X as usize / packed_output_width;
    let packed_column = CUBE_POS_X as usize % packed_output_width;
    let output_column = packed_column * 8;
    let input_width = weight.shape(0);
    let mut sum_0 = 0.0_f32;
    let mut sum_1 = 0.0_f32;
    let mut sum_2 = 0.0_f32;
    let mut sum_3 = 0.0_f32;
    let mut sum_4 = 0.0_f32;
    let mut sum_5 = 0.0_f32;
    let mut sum_6 = 0.0_f32;
    let mut sum_7 = 0.0_f32;
    let mut input_column = lane;

    while input_column < input_width {
        let packed = weight[input_column * weight.stride(0) + packed_column * weight.stride(1)];
        let raw_0 = packed & 15u32;
        let raw_1 = (packed >> 4u32) & 15u32;
        let raw_2 = (packed >> 8u32) & 15u32;
        let raw_3 = (packed >> 12u32) & 15u32;
        let raw_4 = (packed >> 16u32) & 15u32;
        let raw_5 = (packed >> 20u32) & 15u32;
        let raw_6 = (packed >> 24u32) & 15u32;
        let raw_7 = (packed >> 28u32) & 15u32;
        let quant_0 = raw_0 as i32 - (raw_0 >= 8u32) as i32 * 16;
        let quant_1 = raw_1 as i32 - (raw_1 >= 8u32) as i32 * 16;
        let quant_2 = raw_2 as i32 - (raw_2 >= 8u32) as i32 * 16;
        let quant_3 = raw_3 as i32 - (raw_3 >= 8u32) as i32 * 16;
        let quant_4 = raw_4 as i32 - (raw_4 >= 8u32) as i32 * 16;
        let quant_5 = raw_5 as i32 - (raw_5 >= 8u32) as i32 * 16;
        let quant_6 = raw_6 as i32 - (raw_6 >= 8u32) as i32 * 16;
        let quant_7 = raw_7 as i32 - (raw_7 >= 8u32) as i32 * 16;
        let scale_row = input_column / block_size;
        let scale_offset = scale_row * scales.stride(0) + output_column * scales.stride(1);
        let value = f32::cast_from(input[row * input_width + input_column]);

        sum_0 += value * scales[scale_offset] * f32::cast_from(quant_0);
        sum_1 += value * scales[scale_offset + scales.stride(1)] * f32::cast_from(quant_1);
        sum_2 += value * scales[scale_offset + 2 * scales.stride(1)] * f32::cast_from(quant_2);
        sum_3 += value * scales[scale_offset + 3 * scales.stride(1)] * f32::cast_from(quant_3);
        sum_4 += value * scales[scale_offset + 4 * scales.stride(1)] * f32::cast_from(quant_4);
        sum_5 += value * scales[scale_offset + 5 * scales.stride(1)] * f32::cast_from(quant_5);
        sum_6 += value * scales[scale_offset + 6 * scales.stride(1)] * f32::cast_from(quant_6);
        sum_7 += value * scales[scale_offset + 7 * scales.stride(1)] * f32::cast_from(quant_7);
        input_column += CUBE_DIM_X as usize;
    }

    let sum_0 = plane_sum(sum_0);
    let sum_1 = plane_sum(sum_1);
    let sum_2 = plane_sum(sum_2);
    let sum_3 = plane_sum(sum_3);
    let sum_4 = plane_sum(sum_4);
    let sum_5 = plane_sum(sum_5);
    let sum_6 = plane_sum(sum_6);
    let sum_7 = plane_sum(sum_7);
    if lane == 0 {
        let output_offset = row * output_width + output_column;
        output[output_offset] = F::cast_from(sum_0);
        output[output_offset + 1] = F::cast_from(sum_1);
        output[output_offset + 2] = F::cast_from(sum_2);
        output[output_offset + 3] = F::cast_from(sum_3);
        output[output_offset + 4] = F::cast_from(sum_4);
        output[output_offset + 5] = F::cast_from(sum_5);
        output[output_offset + 6] = F::cast_from(sum_6);
        output[output_offset + 7] = F::cast_from(sum_7);
    }
}

#[cube(launch)]
fn q4_gemv_bias_kernel<F: Float>(
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
    let packed_output_width = output_width / 8;
    let row = CUBE_POS_X as usize / packed_output_width;
    let packed_column = CUBE_POS_X as usize % packed_output_width;
    let output_column = packed_column * 8;
    let input_width = weight.shape(0);
    let mut sum_0 = 0.0_f32;
    let mut sum_1 = 0.0_f32;
    let mut sum_2 = 0.0_f32;
    let mut sum_3 = 0.0_f32;
    let mut sum_4 = 0.0_f32;
    let mut sum_5 = 0.0_f32;
    let mut sum_6 = 0.0_f32;
    let mut sum_7 = 0.0_f32;
    let mut input_column = lane;

    while input_column < input_width {
        let packed = weight[input_column * weight.stride(0) + packed_column * weight.stride(1)];
        let raw_0 = packed & 15u32;
        let raw_1 = (packed >> 4u32) & 15u32;
        let raw_2 = (packed >> 8u32) & 15u32;
        let raw_3 = (packed >> 12u32) & 15u32;
        let raw_4 = (packed >> 16u32) & 15u32;
        let raw_5 = (packed >> 20u32) & 15u32;
        let raw_6 = (packed >> 24u32) & 15u32;
        let raw_7 = (packed >> 28u32) & 15u32;
        let quant_0 = raw_0 as i32 - (raw_0 >= 8u32) as i32 * 16;
        let quant_1 = raw_1 as i32 - (raw_1 >= 8u32) as i32 * 16;
        let quant_2 = raw_2 as i32 - (raw_2 >= 8u32) as i32 * 16;
        let quant_3 = raw_3 as i32 - (raw_3 >= 8u32) as i32 * 16;
        let quant_4 = raw_4 as i32 - (raw_4 >= 8u32) as i32 * 16;
        let quant_5 = raw_5 as i32 - (raw_5 >= 8u32) as i32 * 16;
        let quant_6 = raw_6 as i32 - (raw_6 >= 8u32) as i32 * 16;
        let quant_7 = raw_7 as i32 - (raw_7 >= 8u32) as i32 * 16;
        let scale_row = input_column / block_size;
        let scale_offset = scale_row * scales.stride(0) + output_column * scales.stride(1);
        let value = f32::cast_from(input[row * input_width + input_column]);

        sum_0 += value * scales[scale_offset] * f32::cast_from(quant_0);
        sum_1 += value * scales[scale_offset + scales.stride(1)] * f32::cast_from(quant_1);
        sum_2 += value * scales[scale_offset + 2 * scales.stride(1)] * f32::cast_from(quant_2);
        sum_3 += value * scales[scale_offset + 3 * scales.stride(1)] * f32::cast_from(quant_3);
        sum_4 += value * scales[scale_offset + 4 * scales.stride(1)] * f32::cast_from(quant_4);
        sum_5 += value * scales[scale_offset + 5 * scales.stride(1)] * f32::cast_from(quant_5);
        sum_6 += value * scales[scale_offset + 6 * scales.stride(1)] * f32::cast_from(quant_6);
        sum_7 += value * scales[scale_offset + 7 * scales.stride(1)] * f32::cast_from(quant_7);
        input_column += CUBE_DIM_X as usize;
    }

    let sum_0 = plane_sum(sum_0);
    let sum_1 = plane_sum(sum_1);
    let sum_2 = plane_sum(sum_2);
    let sum_3 = plane_sum(sum_3);
    let sum_4 = plane_sum(sum_4);
    let sum_5 = plane_sum(sum_5);
    let sum_6 = plane_sum(sum_6);
    let sum_7 = plane_sum(sum_7);
    if lane == 0 {
        let output_offset = row * output_width + output_column;
        output[output_offset] = F::cast_from(sum_0 + f32::cast_from(bias[output_column]));
        output[output_offset + 1] = F::cast_from(sum_1 + f32::cast_from(bias[output_column + 1]));
        output[output_offset + 2] = F::cast_from(sum_2 + f32::cast_from(bias[output_column + 2]));
        output[output_offset + 3] = F::cast_from(sum_3 + f32::cast_from(bias[output_column + 3]));
        output[output_offset + 4] = F::cast_from(sum_4 + f32::cast_from(bias[output_column + 4]));
        output[output_offset + 5] = F::cast_from(sum_5 + f32::cast_from(bias[output_column + 5]));
        output[output_offset + 6] = F::cast_from(sum_6 + f32::cast_from(bias[output_column + 6]));
        output[output_offset + 7] = F::cast_from(sum_7 + f32::cast_from(bias[output_column + 7]));
    }
}

#[cube(launch)]
fn q4_embedding_kernel<F: Float, I: Int>(
    weight: &Tensor<u32>,
    scales: &Tensor<f32>,
    indices: &Tensor<I>,
    output: &mut Tensor<F>,
    hidden_width: u32,
    #[define(F, I)] _dtypes: [StorageType; 2],
) {
    let index = ABSOLUTE_POS as usize;
    if index >= output.len() {
        terminate!();
    }
    let hidden_width = hidden_width as usize;
    let feature = index % hidden_width;
    let index_row = index / hidden_width;
    let weight_row = usize::cast_from(indices[index_row]);
    let packed_column = feature / 8;
    let packed = weight[weight_row * weight.stride(0) + packed_column * weight.stride(1)];
    let shift = (feature % 8) * 4;
    let raw = (packed >> shift as u32) & 15u32;
    let quant = raw as i32 - (raw >= 8u32) as i32 * 16;
    let scale_column = feature / 32;
    let scale = scales[weight_row * scales.stride(0) + scale_column * scales.stride(1)];
    output[index] = F::cast_from(scale * f32::cast_from(quant));
}

#[cube(launch)]
fn q4_embedding_row_kernel<F: Float>(
    weight: &Tensor<u32>,
    scales: &Tensor<f32>,
    output: &mut Tensor<F>,
    row: u32,
    #[define(F)] _dtype: StorageType,
) {
    let feature = ABSOLUTE_POS as usize;
    if feature >= output.len() {
        terminate!();
    }
    let row = row as usize;
    let packed_column = feature / 8;
    let packed = weight[row * weight.stride(0) + packed_column * weight.stride(1)];
    let shift = (feature % 8) * 4;
    let raw = (packed >> shift as u32) & 15u32;
    let quant = raw as i32 - (raw >= 8u32) as i32 * 16;
    let scale_column = feature / 32;
    let scale = scales[row * scales.stride(0) + scale_column * scales.stride(1)];
    output[feature] = F::cast_from(scale * f32::cast_from(quant));
}

#[cube(launch)]
fn q4_embedding_projection_kernel<F: Float>(
    input: &Tensor<F>,
    weight: &Tensor<u32>,
    scales: &Tensor<f32>,
    output: &mut Tensor<F>,
    semantic_start: u32,
    semantic_rows: u32,
    eos_row: u32,
    #[define(F)] _dtype: StorageType,
) {
    let lane = UNIT_POS_X as usize;
    let semantic_rows = semantic_rows as usize;
    let output_width = semantic_rows + 1;
    let output_index = CUBE_POS_X as usize;
    let input_row = output_index / output_width;
    let projection_column = output_index % output_width;
    let weight_row = if projection_column < semantic_rows {
        semantic_start as usize + projection_column
    } else {
        eos_row as usize
    };
    let hidden_width = weight.shape(1) * 8;
    let mut sum = 0.0_f32;
    let mut feature = lane;

    while feature < hidden_width {
        let packed_column = feature / 8;
        let packed = weight[weight_row * weight.stride(0) + packed_column * weight.stride(1)];
        let shift = (feature % 8) * 4;
        let raw = (packed >> shift as u32) & 15u32;
        let quant = raw as i32 - (raw >= 8u32) as i32 * 16;
        let scale_column = feature / 32;
        let scale = scales[weight_row * scales.stride(0) + scale_column * scales.stride(1)];
        sum += f32::cast_from(input[input_row * hidden_width + feature])
            * scale
            * f32::cast_from(quant);
        feature += CUBE_DIM_X as usize;
    }

    let sum = plane_sum(sum);
    if lane == 0 {
        output[output_index] = F::cast_from(sum);
    }
}

#[cube(launch)]
fn sample_topk_kernel<F: Float, I: Int>(
    logits: &Tensor<F>,
    random_scores: &Tensor<F>,
    random_cursor: &Tensor<I>,
    output: &mut Tensor<I>,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    #[comptime] sort_capacity: u32,
    #[define(F, I)] _dtypes: [StorageType; 2],
) {
    let capacity = sort_capacity as usize;
    let cube_threads = CUBE_DIM_X as usize;
    let planes_per_pass = cube_threads / 32;
    let passes = capacity / cube_threads;
    let mut sorted_values = Shared::new_slice(capacity);
    let mut sorted_indices = Shared::new_slice(capacity);
    let lane = UNIT_POS_X % 32;
    let plane = UNIT_POS_X / 32;

    // Compute the full-softmax denominator in parallel before reusing shared
    // memory for the sorted runs. This preserves the existing top-p policy
    // without making one lane evaluate all 4096 exponentials.
    let mut local_maximum = F::new(-3.0e38_f32);
    let mut reduction_pass = 0usize;
    while reduction_pass < passes {
        let input_index = UNIT_POS_X as usize + reduction_pass * cube_threads;
        if input_index < logits.len() && logits[input_index] > local_maximum {
            local_maximum = logits[input_index];
        }
        reduction_pass += 1;
    }
    let plane_maximum = plane_max(local_maximum);
    if lane == 0 {
        sorted_values[plane as usize] = plane_maximum;
    }
    sync_cube();
    if UNIT_POS_X < 32 {
        let mut partial_maximum = F::new(-3.0e38_f32);
        if (lane as usize) < planes_per_pass {
            partial_maximum = sorted_values[lane as usize];
        }
        let maximum = plane_max(partial_maximum);
        if lane == 0 {
            sorted_values[0] = maximum;
        }
    }
    sync_cube();
    let maximum = sorted_values[0];

    let mut local_sum = 0.0_f32;
    reduction_pass = 0;
    while reduction_pass < passes {
        let input_index = UNIT_POS_X as usize + reduction_pass * cube_threads;
        if input_index < logits.len() {
            local_sum += (f32::cast_from(logits[input_index]) - f32::cast_from(maximum)).exp();
        }
        reduction_pass += 1;
    }
    let partial_plane_sum = plane_sum(local_sum);
    if lane == 0 {
        sorted_values[plane as usize] = F::cast_from(partial_plane_sum);
    }
    sync_cube();
    let mut base_sum = 0.0_f32;
    if UNIT_POS_X < 32 {
        let mut partial_sum = 0.0_f32;
        if (lane as usize) < planes_per_pass {
            partial_sum = f32::cast_from(sorted_values[lane as usize]);
        }
        base_sum = plane_sum(partial_sum);
    }
    // Group zero must finish reading reduction slots before other groups
    // overwrite those locations with their sorted runs.
    sync_cube();

    // Sort every 32-value SIMD-group run entirely in registers. At the
    // production width each thread processes four runs, still with no
    // threadgroup barrier inside the sort.
    let mut pass = 0usize;
    while pass < passes {
        let index = UNIT_POS_X as usize + pass * cube_threads;
        let mut value = F::new(-3.0e38_f32);
        let mut original_index = u32::new(4_294_967_295i64);
        if index < logits.len() {
            value = logits[index];
            original_index = index as u32;
        }
        let mut sequence = 2u32;
        while sequence <= 32 {
            let mut distance = sequence / 2;
            while distance > 0 {
                let partner_value = plane_shuffle_xor(value, distance);
                let partner_index = plane_shuffle_xor(original_index, distance);
                let local_precedes = value > partner_value
                    || (value == partner_value && original_index < partner_index);
                let lower_lane = (lane & distance) == 0;
                let descending_sequence = (lane & sequence) == 0;
                let wants_preceding = lower_lane == descending_sequence;
                if (wants_preceding && !local_precedes) || (!wants_preceding && local_precedes) {
                    value = partner_value;
                    original_index = partner_index;
                }
                distance /= 2;
            }
            sequence *= 2;
        }
        let run = pass * planes_per_pass + plane as usize;
        let run_offset = run * 32;
        sorted_values[run_offset + lane as usize] = value;
        sorted_indices[run_offset + lane as usize] = original_index;
        pass += 1;
    }
    sync_cube();

    // The first plane performs a k-way merge of the sorted runs. Each lane
    // owns up to four run heads, so max/min and cursor advances stay in registers.
    if UNIT_POS_X < 32 {
        let run_count = capacity / 32;
        let invalid_index = u32::new(4_294_967_295i64);
        let mut cursor_0 = 0usize;
        let mut cursor_1 = 0usize;
        let mut cursor_2 = 0usize;
        let mut cursor_3 = 0usize;
        let mut value_0 = F::new(-3.0e38_f32);
        let mut value_1 = F::new(-3.0e38_f32);
        let mut value_2 = F::new(-3.0e38_f32);
        let mut value_3 = F::new(-3.0e38_f32);
        let mut index_0 = invalid_index;
        let mut index_1 = invalid_index;
        let mut index_2 = invalid_index;
        let mut index_3 = invalid_index;
        if (lane as usize) < run_count {
            value_0 = sorted_values[lane as usize * 32];
            index_0 = sorted_indices[lane as usize * 32];
        }
        if (lane as usize) + 32 < run_count {
            value_1 = sorted_values[(lane as usize + 32) * 32];
            index_1 = sorted_indices[(lane as usize + 32) * 32];
        }
        if (lane as usize) + 64 < run_count {
            value_2 = sorted_values[(lane as usize + 64) * 32];
            index_2 = sorted_indices[(lane as usize + 64) * 32];
        }
        if (lane as usize) + 96 < run_count {
            value_3 = sorted_values[(lane as usize + 96) * 32];
            index_3 = sorted_indices[(lane as usize + 96) * 32];
        }

        let maximum = f32::cast_from(maximum);

        let mut cumulative = 0.0_f32;
        let mut keep_len = 0usize;
        let mut sampled_index = 0u32;
        let mut sampled_score = 0.0_f32;
        let random_offset = usize::cast_from(random_cursor[0]);
        let mut rank = 0usize;
        while rank < top_k as usize {
            let mut candidate_value = value_0;
            let mut candidate_index = index_0;
            if value_1 > candidate_value
                || (value_1 == candidate_value && index_1 < candidate_index)
            {
                candidate_value = value_1;
                candidate_index = index_1;
            }
            if value_2 > candidate_value
                || (value_2 == candidate_value && index_2 < candidate_index)
            {
                candidate_value = value_2;
                candidate_index = index_2;
            }
            if value_3 > candidate_value
                || (value_3 == candidate_value && index_3 < candidate_index)
            {
                candidate_value = value_3;
                candidate_index = index_3;
            }
            let next_value = plane_max(candidate_value);
            let mut possible_index = invalid_index;
            if candidate_value == next_value {
                possible_index = candidate_index;
            }
            let next_index = plane_min(possible_index);
            let next_value = f32::cast_from(next_value);
            cumulative += (next_value - maximum).exp() / base_sum;
            if rank > 0 && cumulative > top_p {
                break;
            }
            let score =
                next_value / temperature + f32::cast_from(random_scores[random_offset + rank]);
            if keep_len == 0 || score > sampled_score {
                sampled_score = score;
                sampled_index = next_index;
            }
            keep_len = rank + 1;
            if index_0 == next_index {
                cursor_0 += 1;
                if cursor_0 < 32 {
                    let next_offset = lane as usize * 32 + cursor_0;
                    value_0 = sorted_values[next_offset];
                    index_0 = sorted_indices[next_offset];
                } else {
                    value_0 = F::new(-3.0e38_f32);
                    index_0 = invalid_index;
                }
            }
            if index_1 == next_index {
                cursor_1 += 1;
                if cursor_1 < 32 {
                    let next_offset = (lane as usize + 32) * 32 + cursor_1;
                    value_1 = sorted_values[next_offset];
                    index_1 = sorted_indices[next_offset];
                } else {
                    value_1 = F::new(-3.0e38_f32);
                    index_1 = invalid_index;
                }
            }
            if index_2 == next_index {
                cursor_2 += 1;
                if cursor_2 < 32 {
                    let next_offset = (lane as usize + 64) * 32 + cursor_2;
                    value_2 = sorted_values[next_offset];
                    index_2 = sorted_indices[next_offset];
                } else {
                    value_2 = F::new(-3.0e38_f32);
                    index_2 = invalid_index;
                }
            }
            if index_3 == next_index {
                cursor_3 += 1;
                if cursor_3 < 32 {
                    let next_offset = (lane as usize + 96) * 32 + cursor_3;
                    value_3 = sorted_values[next_offset];
                    index_3 = sorted_indices[next_offset];
                } else {
                    value_3 = F::new(-3.0e38_f32);
                    index_3 = invalid_index;
                }
            }
            rank += 1;
        }
        if lane == 0 {
            output[0] = I::cast_from(sampled_index);
            output[1] = random_cursor[0] + I::cast_from(keep_len as u32);
        }
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
    gemv_prefix_output(input, weight, weight.meta.shape()[1])
}

fn gemv_prefix_output<R: CubeRuntime>(
    input: &CubeTensor<R>,
    weight: &CubeTensor<R>,
    output_width: usize,
) -> CubeTensor<R> {
    assert_eq!(input.meta.num_dims(), 3);
    assert_eq!(weight.meta.num_dims(), 2);
    let input_width = input.meta.shape()[2];
    assert_eq!(weight.meta.shape()[0], input_width);
    assert!(output_width > 0 && output_width <= weight.meta.shape()[1]);
    let mut dims = input.meta.shape().to_vec();
    dims[2] = output_width;
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

    fn fused_grouped_query_attention_decode(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        query.assert_is_on_same_device(&key);
        query.assert_is_on_same_device(&value);
        assert_eq!(query.dtype, key.dtype);
        assert_eq!(query.dtype, value.dtype);
        assert_eq!(query.meta.num_dims(), 4);
        assert_eq!(key.meta.num_dims(), 4);
        assert_eq!(value.meta.num_dims(), 4);
        assert_eq!(query.meta.shape()[0], key.meta.shape()[0]);
        assert_eq!(key.meta.shape(), value.meta.shape());
        assert_eq!(query.meta.shape()[2], 1);
        assert_eq!(query.meta.shape()[3], 64);
        assert_eq!(key.meta.shape()[3], 64);
        assert!(key.meta.shape()[2] > 0);
        assert!(key.meta.shape()[1] > 0);
        assert!(query.meta.shape()[1].is_multiple_of(key.meta.shape()[1]));
        let output = empty_like(&query);
        let cubes = query.meta.shape()[0] * query.meta.shape()[1];
        let dtypes = [
            crate::backend::cubecl::dtype_to_storage_type(query.dtype),
            crate::backend::cubecl::dtype_to_storage_type(DType::F32),
        ];
        if key.meta.shape()[2] < 128 {
            grouped_query_attention_decode_short_kernel::launch(
                &query.client,
                CubeCount::Static(cubes as u32, 1, 1),
                CubeDim::new_1d(32),
                query.clone().into_tensor_arg(),
                key.into_tensor_arg(),
                value.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                dtypes,
            );
            return output;
        }
        let partial_shape = Shape::new([query.meta.shape()[0], query.meta.shape()[1], 8, 66]);
        let partial_handle = query
            .client
            .empty(partial_shape.num_elements() * DType::F32.size());
        let partials = CubeTensor::new_contiguous(
            query.client.clone(),
            query.device.clone(),
            partial_shape,
            partial_handle,
            DType::F32,
        );
        grouped_query_attention_decode_partial_kernel::launch(
            &query.client,
            CubeCount::Static((cubes * 8) as u32, 1, 1),
            CubeDim::new_1d(32),
            query.clone().into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            partials.clone().into_tensor_arg(),
            dtypes,
        );
        grouped_query_attention_decode_merge_kernel::launch(
            &query.client,
            CubeCount::Static(cubes as u32, 1, 1),
            CubeDim::new_1d(32),
            partials.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            dtypes,
        );
        output
    }

    fn fused_gemv(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_width: u32,
    ) -> FloatTensor<Self> {
        input.assert_is_on_same_device(&weight);
        assert_eq!(input.dtype, weight.dtype);
        let input = into_contiguous(input);
        let weight = into_contiguous(weight);
        let output_width = output_width as usize;
        let output = gemv_prefix_output(&input, &weight, output_width);
        gemv_kernel::launch(
            &input.client,
            CubeCount::Static(output.meta.num_elements() as u32, 1, 1),
            CubeDim::new_1d(32),
            input.clone().into_tensor_arg(),
            weight.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            output_width as u32,
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

    fn fused_q8_gemv(
        input: FloatTensor<Self>,
        weight: QuantizedTensor<Self>,
        output_width: u32,
    ) -> FloatTensor<Self> {
        input.assert_is_on_same_device(&weight);
        let scheme = match weight.dtype {
            DType::QFloat(scheme) => scheme,
            dtype => panic!("Expected a quantized Q8 weight, got {dtype:?}"),
        };
        let block_size = q8_gemv_block_size(scheme)
            .expect("Metal Q8 GEMV received an unsupported quantization scheme");
        let input = into_contiguous(input);
        let input_width = weight.meta.shape()[0];
        let full_output_width = weight.meta.shape()[1];
        let output_width = output_width as usize;
        let output = gemv_prefix_output(&input, &weight, output_width);
        assert!(output_width.is_multiple_of(4));

        let (values, scales) = weight
            .quantized_handles()
            .expect("Q8 weight must carry packed values and scales");
        assert_eq!(values.dtype, DType::U32);
        assert_eq!(
            values.meta.shape().dims(),
            [input_width, full_output_width / 4]
        );
        assert_eq!(scales.dtype, DType::F32);
        assert_eq!(
            scales.meta.shape().dims(),
            [input_width, full_output_width.div_ceil(block_size)]
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

    fn fused_q4_gemv(
        input: FloatTensor<Self>,
        weight: QuantizedTensor<Self>,
        output_width: u32,
    ) -> FloatTensor<Self> {
        input.assert_is_on_same_device(&weight);
        let scheme = match weight.dtype {
            DType::QFloat(scheme) => scheme,
            dtype => panic!("Expected a quantized Q4 weight, got {dtype:?}"),
        };
        let block_size = q4_gemv_block_size(scheme)
            .expect("Metal Q4 GEMV received an unsupported quantization scheme");
        let input = into_contiguous(input);
        let input_width = weight.meta.shape()[0];
        let full_output_width = weight.meta.shape()[1];
        let output_width = output_width as usize;
        let output = gemv_prefix_output(&input, &weight, output_width);
        assert!(input_width.is_multiple_of(block_size));
        assert!(output_width.is_multiple_of(8));

        let (values, scales) = weight
            .quantized_handles()
            .expect("Q4 weight must carry packed values and scales");
        assert_eq!(values.dtype, DType::U32);
        assert_eq!(
            values.meta.shape().dims(),
            [input_width, full_output_width / 8]
        );
        assert_eq!(scales.dtype, DType::F32);
        assert_eq!(
            scales.meta.shape().dims(),
            [input_width / block_size, full_output_width]
        );

        let rows = input.meta.num_elements() / input_width;
        q4_gemv_kernel::launch(
            &input.client,
            CubeCount::Static((rows * output_width / 8) as u32, 1, 1),
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

    fn fused_q4_gemv_bias(
        input: FloatTensor<Self>,
        weight: QuantizedTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        input.assert_is_on_same_device(&weight);
        input.assert_is_on_same_device(&bias);
        assert_eq!(input.dtype, bias.dtype);
        let scheme = match weight.dtype {
            DType::QFloat(scheme) => scheme,
            dtype => panic!("Expected a quantized Q4 weight, got {dtype:?}"),
        };
        let block_size = q4_gemv_block_size(scheme)
            .expect("Metal Q4 GEMV received an unsupported quantization scheme");
        let input = into_contiguous(input);
        let bias = into_contiguous(bias);
        let output = gemv_output(&input, &weight);
        let input_width = weight.meta.shape()[0];
        let output_width = weight.meta.shape()[1];
        assert!(input_width.is_multiple_of(block_size));
        assert!(output_width.is_multiple_of(8));
        assert_eq!(bias.meta.num_elements(), output_width);

        let (values, scales) = weight
            .quantized_handles()
            .expect("Q4 weight must carry packed values and scales");
        assert_eq!(values.dtype, DType::U32);
        assert_eq!(values.meta.shape().dims(), [input_width, output_width / 8]);
        assert_eq!(scales.dtype, DType::F32);
        assert_eq!(
            scales.meta.shape().dims(),
            [input_width / block_size, output_width]
        );

        let rows = input.meta.num_elements() / input_width;
        q4_gemv_bias_kernel::launch(
            &input.client,
            CubeCount::Static((rows * output_width / 8) as u32, 1, 1),
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

    fn fused_q4_embedding(
        weight: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        weight.assert_is_on_same_device(&indices);
        let scheme = match weight.dtype {
            DType::QFloat(scheme) => scheme,
            dtype => panic!("Expected a quantized Q4 embedding, got {dtype:?}"),
        };
        assert!(is_q4_embedding_scheme(scheme));
        assert_eq!(weight.meta.num_dims(), 2);
        assert_eq!(indices.meta.num_dims(), 2);
        let rows = weight.meta.shape()[0];
        let hidden_width = weight.meta.shape()[1];
        assert!(hidden_width.is_multiple_of(32));
        let indices = into_contiguous(indices);
        let (values, scales) = weight
            .quantized_handles()
            .expect("Q4 embedding must carry packed values and scales");
        assert_eq!(values.dtype, DType::U32);
        assert_eq!(values.meta.shape().dims(), [rows, hidden_width / 8]);
        assert_eq!(scales.dtype, DType::F32);
        assert_eq!(scales.meta.shape().dims(), [rows, hidden_width / 32]);

        let shape = Shape::new([
            indices.meta.shape()[0],
            indices.meta.shape()[1],
            hidden_width,
        ]);
        let handle = weight
            .client
            .empty(shape.num_elements() * DType::F32.size());
        let output = CubeTensor::new_contiguous(
            weight.client.clone(),
            weight.device.clone(),
            shape,
            handle,
            DType::F32,
        );
        let (cube_count, cube_dim) = elemwise_launch(output.meta.num_elements());
        q4_embedding_kernel::launch(
            &weight.client,
            cube_count,
            cube_dim,
            values.into_tensor_arg(),
            scales.into_tensor_arg(),
            indices.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            hidden_width as u32,
            [
                crate::backend::cubecl::dtype_to_storage_type(DType::F32),
                crate::backend::cubecl::dtype_to_storage_type(indices.dtype),
            ],
        );
        output
    }

    fn fused_q4_embedding_row(weight: QuantizedTensor<Self>, row: u32) -> FloatTensor<Self> {
        let scheme = match weight.dtype {
            DType::QFloat(scheme) => scheme,
            dtype => panic!("Expected a quantized Q4 embedding, got {dtype:?}"),
        };
        assert!(is_q4_embedding_scheme(scheme));
        let rows = weight.meta.shape()[0];
        let hidden_width = weight.meta.shape()[1];
        assert!((row as usize) < rows);
        assert!(hidden_width.is_multiple_of(32));
        let (values, scales) = weight
            .quantized_handles()
            .expect("Q4 embedding must carry packed values and scales");
        assert_eq!(values.meta.shape().dims(), [rows, hidden_width / 8]);
        assert_eq!(scales.meta.shape().dims(), [rows, hidden_width / 32]);

        let shape = Shape::new([1, 1, hidden_width]);
        let handle = weight
            .client
            .empty(shape.num_elements() * DType::F32.size());
        let output = CubeTensor::new_contiguous(
            weight.client.clone(),
            weight.device.clone(),
            shape,
            handle,
            DType::F32,
        );
        let (cube_count, cube_dim) = elemwise_launch(hidden_width);
        q4_embedding_row_kernel::launch(
            &weight.client,
            cube_count,
            cube_dim,
            values.into_tensor_arg(),
            scales.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            row,
            crate::backend::cubecl::dtype_to_storage_type(DType::F32),
        );
        output
    }

    fn fused_q4_embedding_projection(
        input: FloatTensor<Self>,
        weight: QuantizedTensor<Self>,
        semantic_start: u32,
        semantic_rows: u32,
        eos_row: u32,
    ) -> FloatTensor<Self> {
        input.assert_is_on_same_device(&weight);
        let scheme = match weight.dtype {
            DType::QFloat(scheme) => scheme,
            dtype => panic!("Expected a quantized Q4 embedding, got {dtype:?}"),
        };
        assert!(is_q4_embedding_scheme(scheme));
        let table_rows = weight.meta.shape()[0];
        let hidden_width = weight.meta.shape()[1];
        assert_eq!(input.meta.num_dims(), 3);
        assert_eq!(input.meta.shape()[2], hidden_width);
        assert!((semantic_start as usize + semantic_rows as usize) <= table_rows);
        assert!((eos_row as usize) < table_rows);
        let input = into_contiguous(input);
        let (values, scales) = weight
            .quantized_handles()
            .expect("Q4 embedding must carry packed values and scales");
        assert_eq!(values.meta.shape().dims(), [table_rows, hidden_width / 8]);
        assert_eq!(scales.meta.shape().dims(), [table_rows, hidden_width / 32]);

        let input_rows = input.meta.num_elements() / hidden_width;
        let output_width = semantic_rows as usize + 1;
        let shape = Shape::new([input_rows, output_width]);
        let handle = input
            .client
            .empty(shape.num_elements() * input.dtype.size());
        let output = CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            shape,
            handle,
            input.dtype,
        );
        q4_embedding_projection_kernel::launch(
            &input.client,
            CubeCount::Static((input_rows * output_width) as u32, 1, 1),
            CubeDim::new_1d(32),
            input.clone().into_tensor_arg(),
            values.into_tensor_arg(),
            scales.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            semantic_start,
            semantic_rows,
            eos_row,
            crate::backend::cubecl::dtype_to_storage_type(input.dtype),
        );
        output
    }

    fn fused_sample_topk(
        logits: FloatTensor<Self>,
        random_scores: FloatTensor<Self>,
        random_cursor: IntTensor<Self>,
        temperature: f32,
        top_p: f32,
        top_k: u32,
    ) -> IntTensor<Self> {
        logits.assert_is_on_same_device(&random_scores);
        logits.assert_is_on_same_device(&random_cursor);
        assert_eq!(logits.dtype, DType::F32);
        assert_eq!(random_scores.dtype, logits.dtype);
        assert_eq!(logits.meta.num_dims(), 2);
        assert_eq!(logits.meta.shape()[0], 1);
        assert_eq!(random_scores.meta.num_dims(), 1);
        assert_eq!(random_cursor.meta.num_elements(), 1);
        assert!(temperature.is_finite() && temperature > 0.0);
        assert!(top_p.is_finite() && top_p > 0.0 && top_p <= 1.0);
        assert!(top_k > 0 && top_k as usize <= logits.meta.shape()[1]);
        let sort_capacity = logits.meta.shape()[1].next_power_of_two().max(32);
        assert!(
            sort_capacity <= 4_096,
            "Metal residual sampling supports vocabularies up to 4096 entries"
        );

        let logits = into_contiguous(logits);
        let random_scores = into_contiguous(random_scores);
        let random_cursor = into_contiguous(random_cursor);
        let shape = Shape::new([2]);
        let handle = logits
            .client
            .empty(shape.num_elements() * random_cursor.dtype.size());
        let output = CubeTensor::new_contiguous(
            logits.client.clone(),
            logits.device.clone(),
            shape,
            handle,
            random_cursor.dtype,
        );
        sample_topk_kernel::launch(
            &logits.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(sort_capacity.min(1_024) as u32),
            logits.clone().into_tensor_arg(),
            random_scores.into_tensor_arg(),
            random_cursor.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            temperature,
            top_p,
            top_k,
            sort_capacity as u32,
            [
                crate::backend::cubecl::dtype_to_storage_type(logits.dtype),
                crate::backend::cubecl::dtype_to_storage_type(output.dtype),
            ],
        );
        output
    }
}

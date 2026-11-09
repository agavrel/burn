//! Fused operations specialized for Burn's Metal backend.

mod transformer;

pub use transformer::{
    GROUPED_QUERY_ATTENTION_DECODE_HEAD_DIMS, apply_rope, embedding, embedding_projection,
    embedding_row, grouped_query_attention_decode, linear, linear_prefix, rms_norm, sample_topk,
    silu_mul,
};

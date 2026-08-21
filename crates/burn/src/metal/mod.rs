//! Fused operations specialized for Burn's Metal backend.

mod transformer;

pub use transformer::{
    apply_rope, embedding, embedding_projection, embedding_row, linear, linear_prefix, rms_norm,
    sample_topk, silu_mul,
};

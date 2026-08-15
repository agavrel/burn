//! Fused operations specialized for Burn's Metal backend.

mod transformer;

pub use transformer::{apply_rope, linear, rms_norm, silu_mul};

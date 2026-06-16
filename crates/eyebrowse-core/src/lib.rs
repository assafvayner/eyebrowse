//! Shared types and the crate-wide error for the eyebrowse runtime.

/// Logical element type of a [`crate`]-managed tensor.
///
/// Note: f16 weights are physically stored in GPU buffers as packed `u32`
/// (two f16 lanes per word) and unpacked to f32 in-kernel, so [`DType::F16`]
/// describes the logical element, not the storage word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    U32,
    I32,
}

impl DType {
    /// Size in bytes of one logical element.
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::U32 | DType::I32 => 4,
            DType::F16 => 2,
        }
    }
}

/// The single error type used across every eyebrowse crate.
#[derive(thiserror::Error, Debug)]
pub enum EyebrowseError {
    #[error("gpu device unavailable: {0}")]
    DeviceUnavailable(String),
    #[error("load error: {0}")]
    Load(String),
    #[error("unsupported config field: {0}")]
    UnsupportedConfig(String),
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("out of memory: {0}")]
    OutOfMemory(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, EyebrowseError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes() {
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::U32.size(), 4);
        assert_eq!(DType::I32.size(), 4);
        assert_eq!(DType::F16.size(), 2);
    }
}

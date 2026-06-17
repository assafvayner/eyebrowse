//! Decode raw weight bytes (F32/F16/BF16, little-endian, row-major) and upload them to the GPU,
//! either as packed-u32 f16 (for matmul/embedding weights) or as f32 (for RMSNorm weights).

use std::sync::Arc;

use eyebrowse_core::{EyebrowseError, Result};
use eyebrowse_gpu::{Device, Tensor};
use eyebrowse_load::{RawDType, RawTensor};

/// Decode a raw float tensor's bytes to `Vec<f32>`. Integer dtypes are rejected.
pub fn raw_to_f32(raw: &RawTensor) -> Result<Vec<f32>> {
    let out = match raw.dtype {
        RawDType::F32 => raw
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        RawDType::F16 => raw
            .bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        RawDType::BF16 => raw
            .bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        other => {
            return Err(EyebrowseError::UnsupportedConfig(format!(
                "weight dtype {other:?} is not a float"
            )))
        }
    };
    Ok(out)
}

/// Upload a raw float tensor as packed-u32 f16 (the kernels' weight storage). Returns a U32 tensor.
pub fn upload_f16(dev: &Arc<Device>, raw: &RawTensor) -> Result<Tensor> {
    let f = raw_to_f32(raw)?;
    Ok(Tensor::from_f16_packed(dev, &raw.shape, &f))
}

/// Upload a raw float tensor as f32 (used for small RMSNorm weight vectors).
pub fn upload_f32(dev: &Arc<Device>, raw: &RawTensor) -> Result<Tensor> {
    let f = raw_to_f32(raw)?;
    Ok(Tensor::from_f32(dev, &raw.shape, &f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_bf16_roundtrip() {
        // 1.0 and -2.0 as bf16 bits: 0x3F80, 0xC000.
        let bytes = vec![0x80, 0x3F, 0x00, 0xC0];
        let raw = RawTensor {
            bytes,
            dtype: RawDType::BF16,
            shape: vec![2],
        };
        let f = raw_to_f32(&raw).unwrap();
        assert_eq!(f, vec![1.0, -2.0]);
    }
}

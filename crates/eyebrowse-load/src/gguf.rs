//! GGUF v3 weight source: parses the container and exposes tensors via [`WeightSource`].

use std::collections::HashMap;

use eyebrowse_core::{EyebrowseError, Result};

use crate::config::Config;
use crate::dequant::dequant;
use crate::source::{RawDType, RawTensor, WeightSource};

const GGUF_MAGIC: u32 = 0x4655_4747;

/// A scalar metadata value, normalized to the few shapes the config needs.
#[derive(Clone, Debug)]
enum MetaValue {
    U64(u64),
    I64(i64),
    F64(f64),
    String(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    fn as_u64(&self) -> Option<u64> {
        match self {
            MetaValue::U64(v) => Some(*v),
            MetaValue::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    fn as_f32(&self) -> Option<f32> {
        match self {
            MetaValue::F64(v) => Some(*v as f32),
            MetaValue::U64(v) => Some(*v as f32),
            MetaValue::I64(v) => Some(*v as f32),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct TensorInfo {
    ggml_type: u32,
    /// Row-major shape (GGUF's fastest-varying-first dims, reversed).
    shape: Vec<usize>,
    n_elems: usize,
    offset: u64,
}

/// A [`WeightSource`] backed by a single in-memory GGUF v3 file.
pub struct GgufSource {
    config: Config,
    bytes: Vec<u8>,
    data_start: usize,
    tensors: HashMap<String, TensorInfo>,
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    fn need(&self, n: usize) -> Result<()> {
        if self.pos + n > self.data.len() {
            return Err(EyebrowseError::Load("GGUF truncated".to_string()));
        }
        Ok(())
    }

    fn u8(&mut self) -> Result<u8> {
        self.need(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn u32(&mut self) -> Result<u32> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn u64(&mut self) -> Result<u64> {
        self.need(8)?;
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        self.need(len)?;
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .map_err(|e| EyebrowseError::Load(format!("GGUF non-utf8 string: {e}")))?
            .to_string();
        self.pos += len;
        Ok(s)
    }

    /// Read a metadata value of the given type id.
    fn value(&mut self, value_type: u32) -> Result<MetaValue> {
        match value_type {
            0 => Ok(MetaValue::U64(self.u8()? as u64)),
            1 => {
                let v = self.u8()? as i8;
                Ok(MetaValue::I64(v as i64))
            }
            2 => {
                self.need(2)?;
                let v = u16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
                self.pos += 2;
                Ok(MetaValue::U64(v as u64))
            }
            3 => {
                self.need(2)?;
                let v = i16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
                self.pos += 2;
                Ok(MetaValue::I64(v as i64))
            }
            4 => Ok(MetaValue::U64(self.u32()? as u64)),
            5 => Ok(MetaValue::I64(self.u32()? as i32 as i64)),
            6 => {
                let v = f32::from_bits(self.u32()?);
                Ok(MetaValue::F64(v as f64))
            }
            7 => Ok(MetaValue::U64((self.u8()? != 0) as u64)),
            8 => Ok(MetaValue::String(self.string()?)),
            9 => {
                let elem_type = self.u32()?;
                let count = self.u64()? as usize;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.value(elem_type)?);
                }
                Ok(MetaValue::Array(items))
            }
            10 => Ok(MetaValue::U64(self.u64()?)),
            11 => Ok(MetaValue::I64(self.u64()? as i64)),
            12 => {
                let v = f64::from_bits(self.u64()?);
                Ok(MetaValue::F64(v))
            }
            other => Err(EyebrowseError::Load(format!(
                "GGUF unknown metadata value type {other}"
            ))),
        }
    }
}

fn ggml_type_size(ggml_type: u32) -> Result<(usize, usize)> {
    // (block_size_in_elems, block_size_in_bytes)
    match ggml_type {
        0 => Ok((1, 4)),
        1 => Ok((1, 2)),
        8 => Ok((32, 34)),
        12 => Ok((256, 144)),
        14 => Ok((256, 210)),
        other => Err(EyebrowseError::UnsupportedConfig(format!(
            "ggml type {other}"
        ))),
    }
}

impl GgufSource {
    pub fn from_path(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| EyebrowseError::Load(format!("reading {}: {e}", path.display())))?;
        Self::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let (config, data_start, tensors) = parse(&bytes)?;
        Ok(Self {
            config,
            bytes,
            data_start,
            tensors,
        })
    }

    fn has_tensor(&self, gguf_name: &str) -> bool {
        self.tensors.contains_key(gguf_name)
    }
}

/// Map an HF-style tensor name to its GGUF counterpart, or `None` if unrecognized.
fn hf_to_gguf(name: &str) -> Option<String> {
    match name {
        "model.embed_tokens.weight" => return Some("token_embd.weight".to_string()),
        "model.norm.weight" => return Some("output_norm.weight".to_string()),
        "lm_head.weight" => return Some("output.weight".to_string()),
        _ => {}
    }

    let rest = name.strip_prefix("model.layers.")?;
    let (idx, suffix) = rest.split_once('.')?;
    let _: usize = idx.parse().ok()?;
    let suffix = suffix.strip_suffix(".weight")?;

    let target = match suffix {
        "input_layernorm" => "attn_norm",
        "post_attention_layernorm" => "ffn_norm",
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        "self_attn.q_norm" => "attn_q_norm",
        "self_attn.k_norm" => "attn_k_norm",
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        _ => return None,
    };
    Some(format!("blk.{idx}.{target}.weight"))
}

fn parse(bytes: &[u8]) -> Result<(Config, usize, HashMap<String, TensorInfo>)> {
    let mut c = Cursor::new(bytes);
    if c.u32()? != GGUF_MAGIC {
        return Err(EyebrowseError::Load("not a GGUF file".to_string()));
    }
    let version = c.u32()?;
    if version != 3 {
        return Err(EyebrowseError::UnsupportedConfig(format!(
            "GGUF version {version}"
        )));
    }
    let tensor_count = c.u64()? as usize;
    let kv_count = c.u64()? as usize;

    let mut meta: HashMap<String, MetaValue> = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = c.string()?;
        let value_type = c.u32()?;
        let value = c.value(value_type)?;
        meta.insert(key, value);
    }

    let mut tensors: HashMap<String, TensorInfo> = HashMap::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = c.string()?;
        let n_dims = c.u32()? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(c.u64()? as usize);
        }
        let ggml_type = c.u32()?;
        let offset = c.u64()?;
        let n_elems: usize = dims.iter().product();
        // GGUF stores dims fastest-varying first; reverse to row-major.
        let mut shape = dims;
        shape.reverse();
        tensors.insert(
            name,
            TensorInfo {
                ggml_type,
                shape,
                n_elems,
                offset,
            },
        );
    }

    let alignment = meta
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;
    let data_start = c.pos.div_ceil(alignment) * alignment;

    let config = build_config(&meta, &tensors)?;
    Ok((config, data_start, tensors))
}

fn build_config(
    meta: &HashMap<String, MetaValue>,
    tensors: &HashMap<String, TensorInfo>,
) -> Result<Config> {
    let arch = meta
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .ok_or_else(|| EyebrowseError::Load("missing general.architecture".to_string()))?
        .to_string();

    let get_u64 = |suffix: &str| -> Option<usize> {
        meta.get(&format!("{arch}.{suffix}"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    };
    let get_f32 = |suffix: &str| -> Option<f32> {
        meta.get(&format!("{arch}.{suffix}"))
            .and_then(|v| v.as_f32())
    };

    let hidden = get_u64("embedding_length").ok_or_else(|| missing(&arch, "embedding_length"))?;
    let n_layers = get_u64("block_count").ok_or_else(|| missing(&arch, "block_count"))?;
    let n_heads =
        get_u64("attention.head_count").ok_or_else(|| missing(&arch, "attention.head_count"))?;
    let n_kv_heads = get_u64("attention.head_count_kv").unwrap_or(n_heads);
    let head_dim = get_u64("attention.key_length").unwrap_or(hidden / n_heads);
    let intermediate =
        get_u64("feed_forward_length").ok_or_else(|| missing(&arch, "feed_forward_length"))?;
    let rms_eps = get_f32("attention.layer_norm_rms_epsilon")
        .ok_or_else(|| missing(&arch, "attention.layer_norm_rms_epsilon"))?;
    let rope_theta = get_f32("rope.freq_base").unwrap_or(10000.0);
    let max_seq = get_u64("context_length").ok_or_else(|| missing(&arch, "context_length"))?;

    let vocab = get_u64("vocab_size").unwrap_or_else(|| {
        meta.get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                MetaValue::Array(items) => Some(items.len()),
                _ => None,
            })
            .unwrap_or(0)
    });

    let tie_word_embeddings = !tensors.contains_key("output.weight");

    Ok(Config {
        arch,
        hidden,
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        intermediate,
        vocab,
        rms_eps,
        rope_theta,
        tie_word_embeddings,
        max_seq,
    })
}

fn missing(arch: &str, suffix: &str) -> EyebrowseError {
    EyebrowseError::Load(format!("missing {arch}.{suffix}"))
}

impl WeightSource for GgufSource {
    fn config(&self) -> &Config {
        &self.config
    }

    fn tensor_names(&self) -> Vec<String> {
        let n = self.config.n_layers;
        let mut candidates = vec![
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        for i in 0..n {
            for suffix in [
                "input_layernorm",
                "post_attention_layernorm",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "self_attn.q_norm",
                "self_attn.k_norm",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                candidates.push(format!("model.layers.{i}.{suffix}.weight"));
            }
        }
        let mut names: Vec<String> = candidates
            .into_iter()
            .filter(|hf| hf_to_gguf(hf).map(|g| self.has_tensor(&g)).unwrap_or(false))
            .collect();
        names.sort();
        names
    }

    fn raw(&self, name: &str) -> Result<RawTensor> {
        let gguf_name = hf_to_gguf(name)
            .ok_or_else(|| EyebrowseError::Load(format!("unknown tensor {name}")))?;
        let info = self
            .tensors
            .get(&gguf_name)
            .ok_or_else(|| EyebrowseError::Load(format!("unknown tensor {name}")))?;

        let (block_elems, block_bytes) = ggml_type_size(info.ggml_type)?;
        let n_blocks = info.n_elems.div_ceil(block_elems);
        let byte_len = n_blocks * block_bytes;
        let start = self.data_start + info.offset as usize;
        let end = start + byte_len;
        if end > self.bytes.len() {
            return Err(EyebrowseError::Load(format!(
                "tensor {name} data out of bounds"
            )));
        }
        let slice = &self.bytes[start..end];
        let values = dequant(info.ggml_type, slice, info.n_elems)?;
        Ok(RawTensor {
            bytes: bytemuck::cast_slice::<f32, u8>(&values).to_vec(),
            dtype: RawDType::F32,
            shape: info.shape.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn push_kv_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
        push_str(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // value type u32
        buf.extend_from_slice(&val.to_le_bytes());
    }

    fn push_kv_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
        push_str(buf, key);
        buf.extend_from_slice(&6u32.to_le_bytes()); // value type f32
        buf.extend_from_slice(&val.to_bits().to_le_bytes());
    }

    fn push_kv_str(buf: &mut Vec<u8>, key: &str, val: &str) {
        push_str(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // value type string
        push_str(buf, val);
    }

    fn push_tensor_info(buf: &mut Vec<u8>, name: &str, dims: &[u64], ggml_type: u32, offset: u64) {
        push_str(buf, name);
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&ggml_type.to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    fn build_tiny_gguf() -> Vec<u8> {
        // Two F32 tensors: token_embd.weight [4,4] (row-major), output_norm.weight [4].
        let embd: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let norm: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let mut meta = Vec::new();
        let kv_pairs: usize = 10;
        push_kv_str(&mut meta, "general.architecture", "tiny");
        push_kv_u32(&mut meta, "tiny.block_count", 1);
        push_kv_u32(&mut meta, "tiny.embedding_length", 4);
        push_kv_u32(&mut meta, "tiny.attention.head_count", 2);
        push_kv_u32(&mut meta, "tiny.attention.head_count_kv", 1);
        push_kv_u32(&mut meta, "tiny.feed_forward_length", 8);
        push_kv_f32(&mut meta, "tiny.attention.layer_norm_rms_epsilon", 1e-6);
        push_kv_f32(&mut meta, "tiny.rope.freq_base", 10000.0);
        push_kv_u32(&mut meta, "tiny.context_length", 128);
        push_kv_u32(&mut meta, "tiny.vocab_size", 4);

        // Tensor data laid out back to back; offsets relative to data section start.
        let embd_off: u64 = 0;
        let norm_off: u64 = (embd.len() * 4) as u64;

        let mut tinfo = Vec::new();
        // GGUF dims are fastest-varying first; a row-major [4,4] is symmetric, [4] is [4].
        push_tensor_info(&mut tinfo, "token_embd.weight", &[4, 4], 0, embd_off);
        push_tensor_info(&mut tinfo, "output_norm.weight", &[4], 0, norm_off);

        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&2u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&(kv_pairs as u64).to_le_bytes()); // kv count
        buf.extend_from_slice(&meta);
        buf.extend_from_slice(&tinfo);

        // Pad to alignment (default 32).
        let align = 32;
        while buf.len() % align != 0 {
            buf.push(0);
        }
        // Tensor data section.
        for v in &embd {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in &norm {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    #[test]
    fn parses_tiny_gguf_config() {
        let src = GgufSource::from_bytes(build_tiny_gguf()).unwrap();
        let cfg = src.config();
        assert_eq!(cfg.arch, "tiny");
        assert_eq!(cfg.hidden, 4);
        assert_eq!(cfg.n_layers, 1);
        assert_eq!(cfg.n_heads, 2);
        assert_eq!(cfg.n_kv_heads, 1);
        assert_eq!(cfg.head_dim, 2);
        assert_eq!(cfg.intermediate, 8);
        assert_eq!(cfg.vocab, 4);
        assert_eq!(cfg.max_seq, 128);
        assert!((cfg.rope_theta - 10000.0).abs() < 1.0);
        assert!(cfg.tie_word_embeddings); // no output.weight
    }

    #[test]
    fn raw_embedding_is_row_major() {
        let src = GgufSource::from_bytes(build_tiny_gguf()).unwrap();
        let t = src.raw("model.embed_tokens.weight").unwrap();
        assert_eq!(t.dtype, RawDType::F32);
        assert_eq!(t.shape, vec![4, 4]);
        let vals: &[f32] = bytemuck::cast_slice(&t.bytes);
        let expected: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        assert_eq!(vals, expected.as_slice());
    }

    #[test]
    fn raw_norm_resolves() {
        let src = GgufSource::from_bytes(build_tiny_gguf()).unwrap();
        let t = src.raw("model.norm.weight").unwrap();
        assert_eq!(t.shape, vec![4]);
        let vals: &[f32] = bytemuck::cast_slice(&t.bytes);
        assert_eq!(vals, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn unknown_tensor_errors() {
        let src = GgufSource::from_bytes(build_tiny_gguf()).unwrap();
        assert!(src.raw("model.layers.0.mlp.up_proj.weight").is_err());
        assert!(src.raw("nonsense.weight").is_err());
    }

    #[test]
    fn tensor_names_lists_present_only() {
        let src = GgufSource::from_bytes(build_tiny_gguf()).unwrap();
        let names = src.tensor_names();
        assert!(names.contains(&"model.embed_tokens.weight".to_string()));
        assert!(names.contains(&"model.norm.weight".to_string()));
        assert!(!names.contains(&"lm_head.weight".to_string()));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = build_tiny_gguf();
        bytes[0] ^= 0xFF;
        assert!(GgufSource::from_bytes(bytes).is_err());
    }
}

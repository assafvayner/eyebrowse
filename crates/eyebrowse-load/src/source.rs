//! Format-agnostic access to model weights.

use std::collections::HashMap;
use std::path::Path;

use eyebrowse_core::{EyebrowseError, Result};
use safetensors::tensor::{Dtype, Metadata, SafeTensors};
use serde::Deserialize;

use crate::config::{config_from_hf_json, Config};

/// Element type of a raw weight tensor, as stored on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawDType {
    F32,
    F16,
    BF16,
    U8,
    I8,
    I32,
    I64,
}

/// A weight tensor in its on-disk representation: little-endian, row-major bytes.
pub struct RawTensor {
    pub bytes: Vec<u8>,
    pub dtype: RawDType,
    pub shape: Vec<usize>,
}

/// A source of model weights, independent of the underlying file format.
pub trait WeightSource {
    fn config(&self) -> &Config;
    fn tensor_names(&self) -> Vec<String>;
    /// Return the named tensor's bytes (little-endian, row-major), dtype, and shape.
    fn raw(&self, name: &str) -> Result<RawTensor>;
}

fn map_dtype(d: Dtype) -> Result<RawDType> {
    match d {
        Dtype::F32 => Ok(RawDType::F32),
        Dtype::F16 => Ok(RawDType::F16),
        Dtype::BF16 => Ok(RawDType::BF16),
        Dtype::U8 => Ok(RawDType::U8),
        Dtype::I8 => Ok(RawDType::I8),
        Dtype::I32 => Ok(RawDType::I32),
        Dtype::I64 => Ok(RawDType::I64),
        other => Err(EyebrowseError::UnsupportedConfig(format!(
            "safetensors dtype {other:?}"
        ))),
    }
}

#[derive(Deserialize)]
struct IndexFile {
    weight_map: HashMap<String, String>,
}

/// One parsed `.safetensors` shard: the file bytes, the byte offset where the data section begins
/// (`8 + header_len`), and the tensor metadata parsed once. `TensorInfo` offsets are relative to
/// the data section, so the absolute byte range of a tensor is `data_start + offsets`.
struct Shard {
    bytes: Vec<u8>,
    data_start: usize,
    meta: Metadata,
}

impl Shard {
    fn parse(name: &str, bytes: Vec<u8>) -> Result<Self> {
        let (header_len, meta) = SafeTensors::read_metadata(&bytes)
            .map_err(|e| EyebrowseError::Load(format!("parsing {name} header: {e}")))?;
        Ok(Shard {
            data_start: 8 + header_len,
            bytes,
            meta,
        })
    }
}

/// A [`WeightSource`] backed by one or more `.safetensors` files in a directory.
///
/// Each shard's header is parsed exactly once at load time; `raw` then slices the requested tensor
/// directly out of the shard bytes (no per-lookup header re-parse).
pub struct SafeTensorsSource {
    config: Config,
    /// Shard file name -> parsed shard.
    shards: HashMap<String, Shard>,
    /// Tensor name -> shard file name holding it.
    locations: HashMap<String, String>,
}

impl SafeTensorsSource {
    /// Load a model directory: parses `config.json` and the `.safetensors` weights.
    ///
    /// Supports a single `model.safetensors` and a sharded set described by
    /// `model.safetensors.index.json` (`{ "weight_map": { name: shard } }`).
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let config_json = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| EyebrowseError::Load(format!("reading config.json: {e}")))?;
        let config = config_from_hf_json(&config_json)?;

        let index_path = dir.join("model.safetensors.index.json");
        let single_path = dir.join("model.safetensors");

        // The set of shard files to load, regardless of single-file vs. sharded layout.
        let shard_files: Vec<String> = if index_path.exists() {
            let index_json = std::fs::read_to_string(&index_path)
                .map_err(|e| EyebrowseError::Load(format!("reading index.json: {e}")))?;
            let index: IndexFile = serde_json::from_str(&index_json)
                .map_err(|e| EyebrowseError::Load(format!("parsing index.json: {e}")))?;
            let mut files: Vec<String> = index.weight_map.into_values().collect();
            files.sort();
            files.dedup();
            files
        } else if single_path.exists() {
            vec!["model.safetensors".to_string()]
        } else {
            return Err(EyebrowseError::Load(format!(
                "no model.safetensors or model.safetensors.index.json in {}",
                dir.display()
            )));
        };

        let mut shards: HashMap<String, Shard> = HashMap::new();
        let mut locations: HashMap<String, String> = HashMap::new();
        for shard_name in shard_files {
            let bytes = std::fs::read(dir.join(&shard_name))
                .map_err(|e| EyebrowseError::Load(format!("reading shard {shard_name}: {e}")))?;
            let shard = Shard::parse(&shard_name, bytes)?;
            for name in shard.meta.tensors().into_keys() {
                locations.insert(name, shard_name.clone());
            }
            shards.insert(shard_name, shard);
        }

        Ok(Self {
            config,
            shards,
            locations,
        })
    }
}

impl WeightSource for SafeTensorsSource {
    fn config(&self) -> &Config {
        &self.config
    }

    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.locations.keys().cloned().collect();
        names.sort();
        names
    }

    fn raw(&self, name: &str) -> Result<RawTensor> {
        let shard_name = self
            .locations
            .get(name)
            .ok_or_else(|| EyebrowseError::Load(format!("unknown tensor {name}")))?;
        let shard = self
            .shards
            .get(shard_name)
            .ok_or_else(|| EyebrowseError::Load(format!("missing shard {shard_name}")))?;
        let info = shard.meta.info(name).ok_or_else(|| {
            EyebrowseError::Load(format!("tensor {name} not present in shard {shard_name}"))
        })?;
        let start = shard.data_start + info.data_offsets.0;
        let end = shard.data_start + info.data_offsets.1;
        let bytes = shard
            .bytes
            .get(start..end)
            .ok_or_else(|| EyebrowseError::Load(format!("tensor {name}: data offsets out of range")))?
            .to_vec();
        Ok(RawTensor {
            bytes,
            dtype: map_dtype(info.dtype)?,
            shape: info.shape.clone(),
        })
    }
}

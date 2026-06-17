//! Format-agnostic access to model weights.

use std::collections::HashMap;
use std::path::Path;

use eyebrowse_core::{EyebrowseError, Result};
use safetensors::tensor::{Dtype, SafeTensors};
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

/// A [`WeightSource`] backed by one or more `.safetensors` files in a directory.
pub struct SafeTensorsSource {
    config: Config,
    /// Shard file name -> full file contents.
    shards: HashMap<String, Vec<u8>>,
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

        let mut shards: HashMap<String, Vec<u8>> = HashMap::new();
        let mut locations: HashMap<String, String> = HashMap::new();

        if index_path.exists() {
            let index_json = std::fs::read_to_string(&index_path)
                .map_err(|e| EyebrowseError::Load(format!("reading index.json: {e}")))?;
            let index: IndexFile = serde_json::from_str(&index_json)
                .map_err(|e| EyebrowseError::Load(format!("parsing index.json: {e}")))?;

            for (tensor, shard) in index.weight_map {
                if !shards.contains_key(&shard) {
                    let bytes = std::fs::read(dir.join(&shard))
                        .map_err(|e| EyebrowseError::Load(format!("reading shard {shard}: {e}")))?;
                    shards.insert(shard.clone(), bytes);
                }
                locations.insert(tensor, shard);
            }
        } else if single_path.exists() {
            let bytes = std::fs::read(&single_path)
                .map_err(|e| EyebrowseError::Load(format!("reading model.safetensors: {e}")))?;
            let shard = "model.safetensors".to_string();
            let st = SafeTensors::deserialize(&bytes)
                .map_err(|e| EyebrowseError::Load(format!("parsing model.safetensors: {e}")))?;
            for name in st.names() {
                locations.insert(name.to_string(), shard.clone());
            }
            shards.insert(shard, bytes);
        } else {
            return Err(EyebrowseError::Load(format!(
                "no model.safetensors or model.safetensors.index.json in {}",
                dir.display()
            )));
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
        let shard = self
            .locations
            .get(name)
            .ok_or_else(|| EyebrowseError::Load(format!("unknown tensor {name}")))?;
        let bytes = self
            .shards
            .get(shard)
            .ok_or_else(|| EyebrowseError::Load(format!("missing shard {shard}")))?;
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| EyebrowseError::Load(format!("parsing {shard}: {e}")))?;
        let view = st
            .tensor(name)
            .map_err(|e| EyebrowseError::Load(format!("tensor {name}: {e}")))?;
        Ok(RawTensor {
            bytes: view.data().to_vec(),
            dtype: map_dtype(view.dtype())?,
            shape: view.shape().to_vec(),
        })
    }
}

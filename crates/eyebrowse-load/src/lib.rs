//! Model ingestion for the eyebrowse runtime: config, safetensors, and tokenizer parsing.

pub mod config;
pub mod dequant;
pub mod gguf;
pub mod source;
pub mod tokenizer;

pub use config::{config_from_hf_json, Config};
pub use dequant::dequant;
pub use gguf::GgufSource;
pub use source::{RawDType, RawTensor, SafeTensorsSource, WeightSource};
pub use tokenizer::{decode, encode, load_tokenizer};

//! The eyebrowse text-generation runtime: a `Generator` that loads a model + tokenizer and
//! greedily decodes tokens, driving the autoregressive prefill/decode loop over the GPU runtime.

mod generator;

pub use generator::Generator;

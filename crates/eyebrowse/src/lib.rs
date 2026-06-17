//! The eyebrowse text-generation runtime: greedy decoding over the GPU runtime. The native
//! `Generator` adds tokenization; the wasm binding is id-in / id-out (host tokenizes).

mod decode;

pub use decode::greedy_generate;

#[cfg(not(target_arch = "wasm32"))]
mod generator;
#[cfg(not(target_arch = "wasm32"))]
pub use generator::Generator;

#[cfg(target_arch = "wasm32")]
mod wasm;

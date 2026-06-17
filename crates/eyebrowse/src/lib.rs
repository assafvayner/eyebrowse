//! The eyebrowse text-generation runtime: greedy decoding over the GPU runtime. The
//! `Generator` adds tokenization on top of the id-in / id-out decode loop.

mod decode;
mod generator;

pub use decode::greedy_generate;
pub use generator::Generator;

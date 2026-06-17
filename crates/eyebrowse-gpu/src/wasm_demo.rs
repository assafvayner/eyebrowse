//! Phase-0 browser proof: a `#[wasm_bindgen]` entry point that runs the `add` kernel on
//! WebGPU inside the browser and returns the summed result. Compiled only for wasm32.

use std::sync::Arc;

use wasm_bindgen::prelude::*;

use crate::{add, DType, Device, Recorder, Tensor};

/// Initialize a WebGPU device, run `out = a + b` on the GPU, read it back, and return the
/// sum of `out`. Expected value for the baked-in inputs is `231.0`.
#[wasm_bindgen]
pub async fn run_add_demo() -> std::result::Result<f32, JsValue> {
    console_error_panic_hook::set_once();
    let d: Arc<Device> = Device::new()
        .await
        .map_err(|e| JsValue::from_str(&format!("device: {e}")))?;
    let a = Tensor::from_f32(&d, &[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::from_f32(&d, &[6], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let out = Tensor::empty(&d, &[6], DType::F32);
    let mut rec = Recorder::new(&d);
    add(&mut rec, &a, &b, &out);
    rec.submit();
    let got = out
        .to_f32()
        .await
        .map_err(|e| JsValue::from_str(&format!("readback: {e}")))?;
    Ok(got.iter().sum())
}

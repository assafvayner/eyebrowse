//! GPU foundation: wgpu device, tensors over GPU buffers, a command recorder, and the
//! low-level kernel-dispatch helper. Everything above this crate composes these.

mod device;
mod kernel;
mod recorder;
mod tensor;

pub use device::Device;
pub use kernel::{add, copy_range, dispatch, dispatch_with_uniform, uniform_u32};
pub use recorder::Recorder;
pub use tensor::{pack_f16, Tensor};

pub use eyebrowse_core::{DType, EyebrowseError, Result};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn dev() -> Arc<Device> {
        pollster::block_on(Device::new()).expect("device")
    }

    #[test]
    fn device_inits() {
        let _ = dev();
    }

    #[test]
    fn roundtrip() {
        let d = dev();
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_f32(&d, &[2, 3], &data);
        let got = pollster::block_on(t.to_f32()).unwrap();
        assert_eq!(got, data);
    }

    #[test]
    fn add_works() {
        let d = dev();
        let a = Tensor::from_f32(&d, &[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::from_f32(&d, &[6], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let out = Tensor::empty(&d, &[6], DType::F32);
        let mut rec = Recorder::new(&d);
        add(&mut rec, &a, &b, &out);
        rec.submit();
        let got = pollster::block_on(out.to_f32()).unwrap();
        assert_eq!(got, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
    }
}

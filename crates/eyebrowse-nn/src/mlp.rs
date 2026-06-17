use eyebrowse_core::DType;
use eyebrowse_gpu::{Recorder, Tensor};
use eyebrowse_kernels::{geglu, swiglu};

use crate::Linear;

/// Gated activation used by the MLP block.
#[derive(Clone, Copy)]
pub enum Act {
    SwiGlu,
    GeGlu,
}

/// Gated feed-forward block: `down(act(gate(x)) * up(x))`.
pub struct Mlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub act: Act,
}

impl Mlp {
    pub fn new(gate: Linear, up: Linear, down: Linear) -> Self {
        Mlp {
            gate,
            up,
            down,
            act: Act::SwiGlu,
        }
    }

    pub fn geglu(gate: Linear, up: Linear, down: Linear) -> Self {
        Mlp {
            gate,
            up,
            down,
            act: Act::GeGlu,
        }
    }

    /// `x` is `[rows, hidden]` f32; returns `[rows, hidden]` f32.
    pub fn forward(&self, rec: &mut Recorder, x: &Tensor, rows: usize) -> Tensor {
        let inter = self.gate.out_f;
        let g = self.gate.forward(rec, x, rows);
        let u = self.up.forward(rec, x, rows);
        let h = Tensor::empty(rec.device(), &[rows, inter], DType::F32);
        match self.act {
            Act::SwiGlu => swiglu(rec, &g, &u, &h, rows * inter),
            Act::GeGlu => geglu(rec, &g, &u, &h, rows * inter),
        }
        self.down.forward(rec, &h, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eyebrowse_gpu::Device;
    use half::f16;

    fn round_f16(x: &[f32]) -> Vec<f32> {
        x.iter().map(|v| f16::from_f32(*v).to_f32()).collect()
    }

    fn pack_f16(x: &[f32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(x.len().div_ceil(2));
        let mut i = 0;
        while i < x.len() {
            let lo = f16::from_f32(x[i]).to_bits() as u32;
            let hi = if i + 1 < x.len() {
                f16::from_f32(x[i + 1]).to_bits() as u32
            } else {
                0
            };
            out.push(lo | (hi << 16));
            i += 2;
        }
        out
    }

    fn cpu_linear(x: &[f32], w: &[f32], m: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; m * out_f];
        for i in 0..m {
            for o in 0..out_f {
                let mut acc = 0.0f32;
                for k in 0..in_f {
                    acc += x[i * in_f + k] * w[o * in_f + k];
                }
                y[i * out_f + o] = acc;
            }
        }
        y
    }

    fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (x, y) in a.iter().zip(b) {
            let d = (*x as f64) - (*y as f64);
            num += d * d;
            den += (*y as f64) * (*y as f64);
        }
        (num.sqrt() / (den.sqrt() + 1e-12)) as f32
    }

    fn gelu(g: f32) -> f32 {
        0.5 * g * (1.0 + (0.7978846 * (g + 0.044715 * g * g * g)).tanh())
    }

    #[test]
    fn geglu_mlp_matches_cpu() {
        let d = pollster::block_on(Device::new()).expect("device");
        let (rows, hidden, inter) = (3usize, 8usize, 16usize);
        let x: Vec<f32> = (0..rows * hidden).map(|i| i as f32 * 0.03 - 0.4).collect();
        let wg: Vec<f32> = (0..inter * hidden).map(|i| i as f32 * 0.005 - 0.3).collect();
        let wu: Vec<f32> = (0..inter * hidden).map(|i| i as f32 * 0.004 - 0.2).collect();
        let wd: Vec<f32> = (0..hidden * inter).map(|i| i as f32 * 0.003 - 0.25).collect();

        let mk = |w: &[f32], inf, outf| {
            Linear::new(Tensor::from_u32(&d, &[pack_f16(w).len()], &pack_f16(w)), inf, outf)
        };
        let mlp = Mlp::geglu(
            mk(&wg, hidden, inter),
            mk(&wu, hidden, inter),
            mk(&wd, inter, hidden),
        );

        let xt = Tensor::from_f32(&d, &[rows, hidden], &x);
        let mut rec = Recorder::new(&d);
        let out = mlp.forward(&mut rec, &xt, rows);
        rec.submit();
        let got = pollster::block_on(out.to_f32()).unwrap();

        let g = cpu_linear(&x, &round_f16(&wg), rows, hidden, inter);
        let u = cpu_linear(&x, &round_f16(&wu), rows, hidden, inter);
        let h: Vec<f32> = g.iter().zip(u.iter()).map(|(a, b)| gelu(*a) * *b).collect();
        let want = cpu_linear(&h, &round_f16(&wd), rows, inter, hidden);
        assert!(rel_l2(&got, &want) < 2e-3, "rel_l2 = {}", rel_l2(&got, &want));
    }
}

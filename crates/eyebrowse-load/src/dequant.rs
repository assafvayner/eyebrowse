//! Block dequantization for ggml/GGUF weight types.
//!
//! Ports the relevant llama.cpp `dequantize_row_*` routines to f32 output.

use eyebrowse_core::{EyebrowseError, Result};
use half::f16;

const QK_K: usize = 256;

fn read_f16(data: &[u8], off: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([data[off], data[off + 1]])).to_f32()
}

/// Dequantize `n_elems` weights of the given ggml type to f32.
pub fn dequant(ggml_type: u32, data: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    match ggml_type {
        0 => dequant_f32(data, n_elems),
        1 => dequant_f16(data, n_elems),
        8 => dequant_q8_0(data, n_elems),
        12 => dequant_q4_k(data, n_elems),
        14 => dequant_q6_k(data, n_elems),
        other => Err(EyebrowseError::UnsupportedConfig(format!(
            "ggml type {other}"
        ))),
    }
}

fn dequant_f32(data: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    if data.len() < n_elems * 4 {
        return Err(EyebrowseError::Load("F32 buffer too small".to_string()));
    }
    let mut out = Vec::with_capacity(n_elems);
    for i in 0..n_elems {
        let off = i * 4;
        out.push(f32::from_le_bytes([
            data[off],
            data[off + 1],
            data[off + 2],
            data[off + 3],
        ]));
    }
    Ok(out)
}

fn dequant_f16(data: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    if data.len() < n_elems * 2 {
        return Err(EyebrowseError::Load("F16 buffer too small".to_string()));
    }
    let mut out = Vec::with_capacity(n_elems);
    for i in 0..n_elems {
        out.push(read_f16(data, i * 2));
    }
    Ok(out)
}

fn dequant_q8_0(data: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    const BLOCK: usize = 32;
    const BYTES: usize = 34;
    if !n_elems.is_multiple_of(BLOCK) {
        return Err(EyebrowseError::Load(format!(
            "Q8_0 n_elems {n_elems} not a multiple of {BLOCK}"
        )));
    }
    let n_blocks = n_elems / BLOCK;
    if data.len() < n_blocks * BYTES {
        return Err(EyebrowseError::Load("Q8_0 buffer too small".to_string()));
    }
    let mut out = Vec::with_capacity(n_elems);
    for b in 0..n_blocks {
        let base = b * BYTES;
        let d = read_f16(data, base);
        for t in 0..BLOCK {
            let q = data[base + 2 + t] as i8;
            out.push(d * q as f32);
        }
    }
    Ok(out)
}

/// Unpack the 6-bit scale/min pair for sub-block `j` (llama.cpp `get_scale_min_k4`).
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

fn dequant_q4_k(data: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    const BYTES: usize = 144;
    if !n_elems.is_multiple_of(QK_K) {
        return Err(EyebrowseError::Load(format!(
            "Q4_K n_elems {n_elems} not a multiple of {QK_K}"
        )));
    }
    let n_blocks = n_elems / QK_K;
    if data.len() < n_blocks * BYTES {
        return Err(EyebrowseError::Load("Q4_K buffer too small".to_string()));
    }
    let mut out = Vec::with_capacity(n_elems);
    for b in 0..n_blocks {
        let base = b * BYTES;
        let d = read_f16(data, base);
        let dmin = read_f16(data, base + 2);
        let scales = &data[base + 4..base + 16];
        let qs = &data[base + 16..base + 144];
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, scales);
            let d_sc = d * sc as f32;
            let d_min = dmin * m as f32;
            let qbase = 32 * (j / 2);
            let shift = 4 * (j & 1);
            for t in 0..32 {
                let q = (qs[qbase + t] >> shift) & 0xF;
                out.push(d_sc * q as f32 - d_min);
            }
        }
    }
    Ok(out)
}

fn dequant_q6_k(data: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    const BYTES: usize = 210;
    if !n_elems.is_multiple_of(QK_K) {
        return Err(EyebrowseError::Load(format!(
            "Q6_K n_elems {n_elems} not a multiple of {QK_K}"
        )));
    }
    let n_blocks = n_elems / QK_K;
    if data.len() < n_blocks * BYTES {
        return Err(EyebrowseError::Load("Q6_K buffer too small".to_string()));
    }
    let mut out = vec![0.0f32; n_elems];
    for b in 0..n_blocks {
        let base = b * BYTES;
        let ql = &data[base..base + 128];
        let qh = &data[base + 128..base + 192];
        let scales = &data[base + 192..base + 208];
        let d = read_f16(data, base + 208);
        let y = &mut out[b * QK_K..(b + 1) * QK_K];
        for n in (0..QK_K).step_by(128) {
            // Per the llama.cpp reference, the ql/qh/sc cursors advance per 128-block:
            // ql += 64, qh += 32, sc += 8. The scale index also picks up `l/16`.
            let ql_off = n / 2;
            let qh_off = n / 4;
            let sc_off = n / 16;
            for l in 0..32 {
                let is = sc_off + l / 16;
                let q1 =
                    ((ql[ql_off + l] & 0xF) as i32 | (((qh[qh_off + l]) & 3) as i32) << 4) - 32;
                let q2 = ((ql[ql_off + l + 32] & 0xF) as i32
                    | (((qh[qh_off + l] >> 2) & 3) as i32) << 4)
                    - 32;
                let q3 =
                    ((ql[ql_off + l] >> 4) as i32 | (((qh[qh_off + l] >> 4) & 3) as i32) << 4) - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) as i32
                    | (((qh[qh_off + l] >> 6) & 3) as i32) << 4)
                    - 32;
                y[n + l] = d * scales[is] as i8 as f32 * q1 as f32;
                y[n + l + 32] = d * scales[is + 2] as i8 as f32 * q2 as f32;
                y[n + l + 64] = d * scales[is + 4] as i8 as f32 * q3 as f32;
                y[n + l + 96] = d * scales[is + 6] as i8 as f32 * q4 as f32;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (g, w) in got.iter().zip(want.iter()) {
            let d = (*g as f64) - (*w as f64);
            num += d * d;
            den += (*w as f64) * (*w as f64);
        }
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    }

    #[test]
    fn unsupported_type_errors() {
        assert!(dequant(99, &[], 0).is_err());
    }

    #[test]
    fn f32_roundtrips() {
        let vals = [1.0f32, -2.5, 3.25, 0.0];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(dequant(0, &bytes, vals.len()).unwrap(), vals);
    }

    #[test]
    fn f16_roundtrips() {
        let vals = [1.0f32, -2.5, 0.5];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }
        assert_eq!(dequant(1, &bytes, vals.len()).unwrap(), vals);
    }

    #[test]
    fn q8_0_hand_built_block() {
        // d = 2.0, qs = [1, 2, 3, ..., 32] -> w = [2, 4, 6, ..., 64].
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(2.0).to_le_bytes());
        for i in 0..32i8 {
            block.push((i + 1) as u8);
        }
        assert_eq!(block.len(), 34);
        let got = dequant(8, &block, 32).unwrap();
        for (i, v) in got.iter().enumerate() {
            assert_eq!(*v, 2.0 * (i as f32 + 1.0));
        }
    }

    // Reference vectors generated independently with the `gguf` Python library's
    // `dequantize` (the canonical llama.cpp port) over hand-built block bytes, then
    // baked here so there is no runtime Python dependency.
    const Q4K_INPUT: [u8; 144] = [
        102, 42, 174, 39, 202, 212, 30, 40, 131, 134, 201, 76, 18, 204, 117, 95, 19, 106, 177, 8,
        95, 166, 253, 68, 155, 226, 57, 128, 215, 46, 117, 204, 19, 106, 177, 8, 95, 166, 253, 68,
        155, 226, 57, 128, 215, 46, 117, 204, 19, 106, 177, 8, 95, 166, 253, 68, 155, 226, 57, 128,
        215, 46, 117, 204, 19, 106, 177, 8, 95, 166, 253, 68, 155, 226, 57, 128, 215, 46, 117, 204,
        19, 106, 177, 8, 95, 166, 253, 68, 155, 226, 57, 128, 215, 46, 117, 204, 19, 106, 177, 8,
        95, 166, 253, 68, 155, 226, 57, 128, 215, 46, 117, 204, 19, 106, 177, 8, 95, 166, 253, 68,
        155, 226, 57, 128, 215, 46, 117, 204, 19, 106, 177, 8, 95, 166, 253, 68, 155, 226, 57, 128,
        215, 46, 117, 204,
    ];
    const Q4K_EXPECTED: [f32; 256] = [
        1.409637,
        4.908783,
        0.4098816,
        3.909027,
        7.408173,
        2.909271,
        6.408417,
        1.909515,
        5.408661,
        0.9097595,
        4.408905,
        -0.08999634,
        3.409149,
        6.908295,
        2.409393,
        5.908539,
        1.409637,
        4.908783,
        0.4098816,
        3.909027,
        7.408173,
        2.909271,
        6.408417,
        1.909515,
        5.408661,
        0.9097595,
        4.408905,
        -0.08999634,
        3.409149,
        6.908295,
        2.409393,
        5.908539,
        0.8197632,
        5.818542,
        10.81732,
        -0.1799927,
        4.818787,
        9.817566,
        14.81635,
        3.819031,
        8.81781,
        13.81659,
        2.819275,
        7.818054,
        12.81683,
        1.819519,
        6.818298,
        11.81708,
        0.8197632,
        5.818542,
        10.81732,
        -0.1799927,
        4.818787,
        9.817566,
        14.81635,
        3.819031,
        8.81781,
        13.81659,
        2.819275,
        7.818054,
        12.81683,
        1.819519,
        6.818298,
        11.81708,
        4.228912,
        14.72635,
        1.229645,
        11.72708,
        22.22452,
        8.727814,
        19.22525,
        5.728546,
        16.22598,
        2.729279,
        13.22672,
        -0.269989,
        10.22745,
        20.72488,
        7.22818,
        17.72562,
        4.228912,
        14.72635,
        1.229645,
        11.72708,
        22.22452,
        8.727814,
        19.22525,
        5.728546,
        16.22598,
        2.729279,
        13.22672,
        -0.269989,
        10.22745,
        20.72488,
        7.22818,
        17.72562,
        1.639526,
        11.63708,
        21.63464,
        -0.3599854,
        9.637573,
        19.63513,
        29.63269,
        7.638062,
        17.63562,
        27.63318,
        5.63855,
        15.63611,
        25.63367,
        3.639038,
        13.6366,
        23.63416,
        1.639526,
        11.63708,
        21.63464,
        -0.3599854,
        9.637573,
        19.63513,
        29.63269,
        7.638062,
        17.63562,
        27.63318,
        5.63855,
        15.63611,
        25.63367,
        3.639038,
        13.6366,
        23.63416,
        6.508209,
        24.00394,
        1.50943,
        19.00516,
        36.50089,
        14.00638,
        31.50211,
        9.007599,
        26.50333,
        4.00882,
        21.50455,
        -0.9899597,
        16.50577,
        34.0015,
        11.50699,
        29.00272,
        6.508209,
        24.00394,
        1.50943,
        19.00516,
        36.50089,
        14.00638,
        31.50211,
        9.007599,
        26.50333,
        4.00882,
        21.50455,
        -0.9899597,
        16.50577,
        34.0015,
        11.50699,
        29.00272,
        1.679321,
        16.67566,
        31.672,
        -1.319946,
        13.67639,
        28.67273,
        43.66907,
        10.67712,
        25.67346,
        40.6698,
        7.677856,
        22.67419,
        37.67053,
        4.678589,
        19.67493,
        34.67126,
        1.679321,
        16.67566,
        31.672,
        -1.319946,
        13.67639,
        28.67273,
        43.66907,
        10.67712,
        25.67346,
        40.6698,
        7.677856,
        22.67419,
        37.67053,
        4.678589,
        19.67493,
        34.67126,
        -0.900116,
        0.8494568,
        -1.399994,
        0.3495789,
        2.099152,
        -0.1502991,
        1.599274,
        -0.650177,
        1.099396,
        -1.150055,
        0.5995178,
        -1.649933,
        0.09963989,
        1.849213,
        -0.400238,
        1.349335,
        -0.900116,
        0.8494568,
        -1.399994,
        0.3495789,
        2.099152,
        -0.1502991,
        1.599274,
        -0.650177,
        1.099396,
        -1.150055,
        0.5995178,
        -1.649933,
        0.09963989,
        1.849213,
        -0.400238,
        1.349335,
        0.1198425,
        3.868927,
        7.618011,
        -0.6299744,
        3.11911,
        6.868195,
        10.61728,
        2.369293,
        6.118378,
        9.867462,
        1.619476,
        5.368561,
        9.117645,
        0.8696594,
        4.618744,
        8.367828,
        0.1198425,
        3.868927,
        7.618011,
        -0.6299744,
        3.11911,
        6.868195,
        10.61728,
        2.369293,
        6.118378,
        9.867462,
        1.619476,
        5.368561,
        9.117645,
        0.8696594,
        4.618744,
        8.367828,
    ];

    const Q6K_INPUT: [u8; 210] = [
        2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97, 102, 107,
        112, 117, 122, 127, 132, 137, 142, 147, 152, 157, 162, 167, 172, 177, 182, 187, 192, 197,
        202, 207, 212, 217, 222, 227, 232, 237, 242, 247, 252, 1, 6, 11, 16, 21, 26, 31, 36, 41,
        46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116, 121, 126, 131, 136, 141,
        146, 151, 156, 161, 166, 171, 176, 181, 186, 191, 196, 201, 206, 211, 216, 221, 226, 231,
        236, 241, 246, 251, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,
        90, 95, 100, 105, 110, 115, 120, 125, 7, 16, 25, 34, 43, 52, 61, 70, 79, 88, 97, 106, 115,
        124, 133, 142, 151, 160, 169, 178, 187, 196, 205, 214, 223, 232, 241, 250, 3, 12, 21, 30,
        39, 48, 57, 66, 75, 84, 93, 102, 111, 120, 129, 138, 147, 156, 165, 174, 183, 192, 201,
        210, 219, 228, 237, 246, 255, 8, 17, 26, 35, 44, 53, 62, 247, 2, 244, 255, 10, 252, 7, 249,
        4, 246, 1, 12, 254, 9, 251, 6, 31, 37,
    ];
    const Q6K_EXPECTED: [f32; 256] = [
        -3.240692, 4.500961, 0.7201538, -0.1800385, -3.960846, 3.780807, 2.880615, -0.9001923,
        -4.681, 3.060654, 2.160461, -1.620346, -5.401154, 5.221115, 1.440308, -2.3405, 0.7201538,
        -1.000214, -0.1600342, 0.04000854, 0.880188, -0.8401794, -0.6401367, 0.2000427, 1.040222,
        -0.6801453, -0.4801025, 0.3600769, 1.200256, -1.160248, -0.3200684, 0.5201111, 3.360718,
        6.001282, -2.880615, 7.441589, -1.440308, 1.200256, -3.84082, 2.640564, -6.241333,
        -3.600769, 6.721436, -2.160461, 4.320923, -4.560974, 1.92041, -6.961487, 0.2800598,
        0.5001068, -0.2400513, 0.6201324, -0.1200256, 0.1000214, -0.3200684, 0.220047, -0.5201111,
        -0.3000641, 0.5601196, -0.1800385, 0.3600769, -0.3800812, 0.1600342, -0.5801239, -6.401367,
        -3.200684, -3.200684, 0.2000427, 0.2000427, 3.400726, 3.600769, -6.001282, -6.001282,
        -2.800598, 0.6001282, 0.6001282, 3.800812, 4.000854, -5.601196, -5.601196, 0.880188,
        -0.4000854, -0.4000854, -1.760376, -1.760376, 2.080444, 2.000427, 0.7201538, 0.7201538,
        -0.5601196, -1.92041, -1.92041, 1.92041, 1.840393, 0.5601196, 0.5601196, -3.080658,
        -3.080658, -3.080658, -2.940628, -2.940628, -2.940628, -2.800598, -0.5601196, -0.5601196,
        -0.5601196, -0.4200897, -0.4200897, -0.4200897, -0.2800598, 1.960419, 1.960419, -2.100449,
        -2.100449, -2.100449, -0.0, -0.0, -2.240479, -2.380508, -2.380508, -2.380508, -2.380508,
        -2.520538, -2.520538, 4.200897, 4.060867, 4.060867, 4.060867, 1.440308, -2.000427,
        -0.3200684, 0.08001709, 1.760376, -1.680359, -1.280273, 0.4000854, 2.080444, -1.360291,
        -0.9602051, 0.7201538, 2.400513, -2.320496, -0.6401367, 1.040222, -3.600769, 5.001068,
        0.8001709, -0.2000427, -4.40094, 4.200897, 3.200684, -1.000214, -5.201111, 3.400726,
        2.400513, -1.800385, -6.001282, 5.801239, 1.600342, -2.600555, -0.2800598, -0.5001068,
        0.2400513, -0.6201324, 0.1200256, -0.1000214, 0.3200684, -0.220047, 0.5201111, 0.3000641,
        -0.5601196, 0.1800385, -0.3600769, 0.3800812, -0.1600342, 0.5801239, -3.360718, -6.001282,
        2.880615, -7.441589, 1.440308, -1.200256, 3.84082, -2.640564, 6.241333, 3.600769,
        -6.721436, 2.160461, -4.320923, 4.560974, -1.92041, 6.961487, -0.1600342, -0.8001709,
        -0.8001709, 1.080231, 1.080231, 0.440094, 0.4000854, -0.2400513, -0.2400513, -0.880188,
        1.000214, 1.000214, 0.3600769, 0.3200684, -0.3200684, -0.3200684, 4.500961, -4.140884,
        -4.140884, -1.080231, -1.080231, 1.800385, 1.980423, 4.861038, 4.861038, -3.780807,
        -0.7201538, -0.7201538, 2.160461, 2.3405, 5.221115, 5.221115, 1.800385, 1.800385, 1.800385,
        0.1000214, 0.1000214, 0.1000214, 1.600342, 1.600342, 1.600342, 1.600342, -0.1000214,
        -0.1000214, -0.1000214, -0.2000427, -0.2000427, -0.2000427, 0.3600769, 2.280487, 2.280487,
        2.400513, 2.400513, 2.400513, 2.520538, 2.520538, 2.520538, -3.240692, -3.120667,
        -3.120667, -3.120667, -3.000641, -3.000641, -3.000641,
    ];

    #[test]
    fn q4_k_matches_reference() {
        let got = dequant(12, &Q4K_INPUT, 256).unwrap();
        assert!(rel_l2(&got, &Q4K_EXPECTED) < 1e-3);
    }

    #[test]
    fn q6_k_matches_reference() {
        let got = dequant(14, &Q6K_INPUT, 256).unwrap();
        assert!(rel_l2(&got, &Q6K_EXPECTED) < 1e-3);
    }
}

/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Random / Noise generation — port of GF_Op_ random functions.
 */

use crate::matrix::create_matrix;
use crate::types::{NoiseType, TMatrix, TVector};
use rand::Rng;

pub fn random_gaussian() -> f32 {
    let mut rng = rand::thread_rng();
    // Box-Muller transform
    let u1: f32 = rng.gen::<f32>().max(1e-10);
    let u2: f32 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

pub fn random_uniform(lo: f32, hi: f32) -> f32 {
    let mut rng = rand::thread_rng();
    lo + rng.gen::<f32>() * (hi - lo)
}

pub fn random_analog() -> f32 {
    let base = random_gaussian();
    let noise = random_uniform(-0.05, 0.05);
    base + noise
}

pub fn generate_noise(noise: &mut TMatrix, size: i32, depth: i32, nt: NoiseType) {
    *noise = create_matrix(size, depth);
    for i in 0..size as usize {
        for j in 0..depth as usize {
            noise[i][j] = match nt {
                NoiseType::Gauss => random_gaussian(),
                NoiseType::Uniform => random_uniform(-1.0, 1.0),
                NoiseType::Analog => random_analog(),
            };
        }
    }
}

pub fn noise_slerp(v1: &TVector, v2: &TVector, t: f32) -> TVector {
    let n = v1.len();
    let mut result = vec![0.0f32; n];

    let mut dot = 0.0f32;
    let mut norm1 = 0.0f32;
    let mut norm2 = 0.0f32;
    for i in 0..n {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    norm1 = norm1.sqrt();
    norm2 = norm2.sqrt();

    if norm1 < 1e-12 || norm2 < 1e-12 {
        for i in 0..n {
            result[i] = v1[i] * (1.0 - t) + v2[i] * t;
        }
        return result;
    }

    let cos_omega = (dot / (norm1 * norm2)).clamp(-1.0, 1.0);
    let omega = cos_omega.acos();

    if omega.abs() < 1e-6 {
        for i in 0..n {
            result[i] = v1[i] * (1.0 - t) + v2[i] * t;
        }
        return result;
    }

    let sin_omega = omega.sin();
    let s1 = ((1.0 - t) * omega).sin() / sin_omega;
    let s2 = (t * omega).sin() / sin_omega;

    for i in 0..n {
        result[i] = s1 * v1[i] + s2 * v2[i];
    }
    result
}

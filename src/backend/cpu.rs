/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * CPU backend — pure Rust implementation of all acceleratable ops.
 */

use super::BackendOps;

pub struct CpuBackend;

impl BackendOps for CpuBackend {
    fn name(&self) -> &str { "CPU" }

    fn matrix_multiply(&self, a: &[f32], b: &[f32], m: i32, k: i32, n: i32) -> Vec<f32> {
        let (m, k, n) = (m as usize, k as usize, n as usize);
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn matrix_add(&self, a: &[f32], b: &[f32], len: i32) -> Vec<f32> {
        (0..len as usize).map(|i| a[i] + b[i]).collect()
    }

    fn matrix_sub(&self, a: &[f32], b: &[f32], len: i32) -> Vec<f32> {
        (0..len as usize).map(|i| a[i] - b[i]).collect()
    }

    fn matrix_scale(&self, a: &[f32], s: f32, len: i32) -> Vec<f32> {
        (0..len as usize).map(|i| a[i] * s).collect()
    }

    fn matrix_element_mul(&self, a: &[f32], b: &[f32], len: i32) -> Vec<f32> {
        (0..len as usize).map(|i| a[i] * b[i]).collect()
    }

    fn matrix_add_inplace(&self, a: &mut [f32], b: &[f32], len: i32) {
        for i in 0..len as usize { a[i] += b[i]; }
    }

    fn matrix_scale_inplace(&self, a: &mut [f32], s: f32, len: i32) {
        for i in 0..len as usize { a[i] *= s; }
    }

    fn matrix_clip_inplace(&self, a: &mut [f32], lo: f32, hi: f32, len: i32) {
        for i in 0..len as usize {
            if a[i] < lo { a[i] = lo; }
            else if a[i] > hi { a[i] = hi; }
        }
    }

    fn relu_forward(&self, a: &[f32], len: i32) -> Vec<f32> {
        (0..len as usize).map(|i| if a[i] > 0.0 { a[i] } else { 0.0 }).collect()
    }

    fn leaky_relu_forward(&self, a: &[f32], alpha: f32, len: i32) -> Vec<f32> {
        (0..len as usize).map(|i| if a[i] > 0.0 { a[i] } else { alpha * a[i] }).collect()
    }

    fn sigmoid_forward(&self, a: &[f32], len: i32) -> Vec<f32> {
        (0..len as usize).map(|i| {
            let v = a[i].clamp(-88.0, 88.0);
            1.0 / (1.0 + (-v).exp())
        }).collect()
    }

    fn tanh_forward(&self, a: &[f32], len: i32) -> Vec<f32> {
        (0..len as usize).map(|i| a[i].tanh()).collect()
    }

    fn activation_backward(&self, grad: &[f32], pre_act: &[f32], act_type: i32, len: i32)
        -> Vec<f32>
    {
        (0..len as usize).map(|i| {
            let g = grad[i];
            let p = pre_act[i];
            match act_type {
                0 => if p > 0.0 { g } else { 0.0 },           // ReLU
                1 => if p > 0.0 { g } else { 0.01 * g },      // LeakyReLU
                2 => {                                          // Sigmoid
                    let s = 1.0 / (1.0 + (-p).exp());
                    g * s * (1.0 - s)
                }
                3 => {                                          // Tanh
                    let s = p.tanh();
                    g * (1.0 - s * s)
                }
                _ => g,                                         // None
            }
        }).collect()
    }

    fn adam_update(
        &self, p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32],
        t: i32, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, len: i32,
    ) {
        for i in 0..len as usize {
            let grad = g[i] + wd * p[i];
            m[i] = b1 * m[i] + (1.0 - b1) * grad;
            v[i] = b2 * v[i] + (1.0 - b2) * grad * grad;
            let m_hat = m[i] / (1.0 - b1.powi(t));
            let v_hat = v[i] / (1.0 - b2.powi(t));
            p[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    fn sgd_update(&self, p: &mut [f32], g: &[f32], lr: f32, wd: f32, len: i32) {
        for i in 0..len as usize {
            p[i] -= lr * (g[i] + wd * p[i]);
        }
    }

    fn rmsprop_update(
        &self, p: &mut [f32], g: &[f32], cache: &mut [f32],
        lr: f32, decay: f32, eps: f32, wd: f32, len: i32,
    ) {
        for i in 0..len as usize {
            let grad = g[i] + wd * p[i];
            cache[i] = decay * cache[i] + (1.0 - decay) * grad * grad;
            p[i] -= lr * grad / (cache[i].sqrt() + eps);
        }
    }

    fn bce_gradient(&self, pred: &[f32], target: &[f32], len: i32) -> Vec<f32> {
        (0..len as usize).map(|i| {
            let p = pred[i].clamp(1e-7, 1.0 - 1e-7);
            -(target[i] / p - (1.0 - target[i]) / (1.0 - p))
        }).collect()
    }
}

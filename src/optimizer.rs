/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Optimizer update functions — port of GF_Train_ optimizer functions.
 */

use crate::matrix::create_matrix;
use crate::types::TMatrix;

pub fn adam_update_matrix(
    p: &mut TMatrix,
    g: &TMatrix,
    m: &mut TMatrix,
    v: &mut TMatrix,
    t: i32,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
) {
    if p.is_empty() || g.is_empty() {
        return;
    }
    // Ensure m and v are properly sized
    if m.is_empty() {
        *m = create_matrix(p.len() as i32, p[0].len() as i32);
    }
    if v.is_empty() {
        *v = create_matrix(p.len() as i32, p[0].len() as i32);
    }
    let rows = p.len().min(g.len()).min(m.len()).min(v.len());
    for i in 0..rows {
        let cols = p[i].len().min(g[i].len()).min(m[i].len()).min(v[i].len());
        for j in 0..cols {
            let grad = g[i][j] + wd * p[i][j];
            m[i][j] = b1 * m[i][j] + (1.0 - b1) * grad;
            v[i][j] = b2 * v[i][j] + (1.0 - b2) * grad * grad;
            let m_hat = m[i][j] / (1.0 - b1.powi(t));
            let v_hat = v[i][j] / (1.0 - b2.powi(t));
            p[i][j] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

pub fn sgd_update_matrix(p: &mut TMatrix, g: &TMatrix, lr: f32, wd: f32) {
    if p.is_empty() || g.is_empty() {
        return;
    }
    for i in 0..p.len().min(g.len()) {
        for j in 0..p[i].len().min(g[i].len()) {
            p[i][j] -= lr * (g[i][j] + wd * p[i][j]);
        }
    }
}

pub fn rmsprop_update_matrix(
    p: &mut TMatrix,
    g: &TMatrix,
    cache: &mut TMatrix,
    lr: f32,
    decay: f32,
    eps: f32,
    wd: f32,
) {
    if p.is_empty() || g.is_empty() {
        return;
    }
    if cache.is_empty() {
        *cache = create_matrix(p.len() as i32, p[0].len() as i32);
    }
    for i in 0..p.len().min(g.len()).min(cache.len()) {
        for j in 0..p[i].len().min(g[i].len()).min(cache[i].len()) {
            let grad = g[i][j] + wd * p[i][j];
            cache[i][j] = decay * cache[i][j] + (1.0 - decay) * grad * grad;
            p[i][j] -= lr * grad / (cache[i][j].sqrt() + eps);
        }
    }
}

pub fn cosine_anneal(epoch: i32, max_ep: i32, base_lr: f32, min_lr: f32) -> f32 {
    if max_ep <= 0 {
        return base_lr;
    }
    min_lr
        + 0.5 * (base_lr - min_lr)
            * (1.0 + (std::f32::consts::PI * epoch as f32 / max_ep as f32).cos())
}

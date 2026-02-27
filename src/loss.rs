/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Loss functions — port of GF_Train_ loss functions.
 */

use crate::matrix::create_matrix;
use crate::types::TMatrix;

pub fn binary_cross_entropy(pred: &TMatrix, target: &TMatrix) -> f32 {
    let mut loss = 0.0f32;
    let mut count = 0;
    for i in 0..pred.len().min(target.len()) {
        for j in 0..pred[i].len().min(target[i].len()) {
            let p = pred[i][j].clamp(1e-7, 1.0 - 1e-7);
            loss -= target[i][j] * p.ln() + (1.0 - target[i][j]) * (1.0 - p).ln();
            count += 1;
        }
    }
    if count > 0 {
        loss / count as f32
    } else {
        0.0
    }
}

pub fn bce_gradient(pred: &TMatrix, target: &TMatrix) -> TMatrix {
    let rows = pred.len();
    if rows == 0 {
        return vec![];
    }
    let cols = pred[0].len();
    let mut grad = create_matrix(rows as i32, cols as i32);
    for i in 0..rows.min(target.len()) {
        for j in 0..cols.min(target[i].len()) {
            let p = pred[i][j].clamp(1e-7, 1.0 - 1e-7);
            grad[i][j] = -(target[i][j] / p - (1.0 - target[i][j]) / (1.0 - p));
        }
    }
    grad
}

pub fn wgan_disc_loss(d_real: &TMatrix, d_fake: &TMatrix) -> f32 {
    let mut real_mean = 0.0f32;
    let mut fake_mean = 0.0f32;
    let mut count = 0;
    for i in 0..d_real.len() {
        for j in 0..d_real[i].len() {
            real_mean += d_real[i][j];
            count += 1;
        }
    }
    let mut fcount = 0;
    for i in 0..d_fake.len() {
        for j in 0..d_fake[i].len() {
            fake_mean += d_fake[i][j];
            fcount += 1;
        }
    }
    if count > 0 {
        real_mean /= count as f32;
    }
    if fcount > 0 {
        fake_mean /= fcount as f32;
    }
    // WGAN disc loss: -(E[D(real)] - E[D(fake)])
    -(real_mean - fake_mean)
}

pub fn wgan_gen_loss(d_fake: &TMatrix) -> f32 {
    let mut mean = 0.0f32;
    let mut count = 0;
    for i in 0..d_fake.len() {
        for j in 0..d_fake[i].len() {
            mean += d_fake[i][j];
            count += 1;
        }
    }
    if count > 0 {
        -mean / count as f32
    } else {
        0.0
    }
}

pub fn wgan_disc_grad(d_out: &TMatrix, is_real: bool) -> TMatrix {
    let rows = d_out.len();
    if rows == 0 {
        return vec![];
    }
    let cols = d_out[0].len();
    let mut grad = create_matrix(rows as i32, cols as i32);
    let sign = if is_real { -1.0f32 } else { 1.0f32 };
    let n = (rows * cols) as f32;
    for i in 0..rows {
        for j in 0..cols {
            grad[i][j] = sign / n;
        }
    }
    grad
}

pub fn wgan_gen_grad(d_fake: &TMatrix) -> TMatrix {
    let rows = d_fake.len();
    if rows == 0 {
        return vec![];
    }
    let cols = d_fake[0].len();
    let mut grad = create_matrix(rows as i32, cols as i32);
    let n = (rows * cols) as f32;
    for i in 0..rows {
        for j in 0..cols {
            grad[i][j] = -1.0 / n;
        }
    }
    grad
}

pub fn hinge_disc_loss(d_real: &TMatrix, d_fake: &TMatrix) -> f32 {
    let mut loss = 0.0f32;
    let mut count = 0;
    for i in 0..d_real.len() {
        for j in 0..d_real[i].len() {
            loss += (1.0 - d_real[i][j]).max(0.0);
            count += 1;
        }
    }
    for i in 0..d_fake.len() {
        for j in 0..d_fake[i].len() {
            loss += (1.0 + d_fake[i][j]).max(0.0);
            count += 1;
        }
    }
    if count > 0 {
        loss / count as f32
    } else {
        0.0
    }
}

pub fn hinge_gen_loss(d_fake: &TMatrix) -> f32 {
    let mut mean = 0.0f32;
    let mut count = 0;
    for i in 0..d_fake.len() {
        for j in 0..d_fake[i].len() {
            mean += d_fake[i][j];
            count += 1;
        }
    }
    if count > 0 {
        -mean / count as f32
    } else {
        0.0
    }
}

pub fn ls_disc_loss(d_real: &TMatrix, d_fake: &TMatrix) -> f32 {
    let mut loss = 0.0f32;
    let mut count = 0;
    for i in 0..d_real.len() {
        for j in 0..d_real[i].len() {
            let d = d_real[i][j] - 1.0;
            loss += d * d;
            count += 1;
        }
    }
    for i in 0..d_fake.len() {
        for j in 0..d_fake[i].len() {
            loss += d_fake[i][j] * d_fake[i][j];
            count += 1;
        }
    }
    if count > 0 {
        0.5 * loss / count as f32
    } else {
        0.0
    }
}

pub fn ls_gen_loss(d_fake: &TMatrix) -> f32 {
    let mut loss = 0.0f32;
    let mut count = 0;
    for i in 0..d_fake.len() {
        for j in 0..d_fake[i].len() {
            let d = d_fake[i][j] - 1.0;
            loss += d * d;
            count += 1;
        }
    }
    if count > 0 {
        0.5 * loss / count as f32
    } else {
        0.0
    }
}

pub fn apply_label_smoothing(labels: &TMatrix, lo: f32, hi: f32) -> TMatrix {
    let rows = labels.len();
    if rows == 0 {
        return vec![];
    }
    let cols = labels[0].len();
    let mut result = create_matrix(rows as i32, cols as i32);
    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = if labels[i][j] > 0.5 { hi } else { lo };
        }
    }
    result
}

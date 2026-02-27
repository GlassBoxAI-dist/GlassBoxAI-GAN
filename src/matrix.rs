/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Matrix operations — port of GF_Op_ matrix functions.
 */

use crate::types::{TMatrix, TVector};

pub fn create_matrix(rows: i32, cols: i32) -> TMatrix {
    vec![vec![0.0f32; cols as usize]; rows as usize]
}

pub fn create_vector(size: i32) -> TVector {
    vec![0.0f32; size as usize]
}

pub fn matrix_multiply(a: &TMatrix, b: &TMatrix) -> TMatrix {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let rows = a.len();
    let cols = b[0].len();
    let inner = a[0].len();
    let mut result = create_matrix(rows as i32, cols as i32);
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0f32;
            for k in 0..inner {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    result
}

pub fn matrix_add(a: &TMatrix, b: &TMatrix) -> TMatrix {
    let rows = a.len();
    if rows == 0 {
        return vec![];
    }
    let cols = a[0].len();
    let mut result = create_matrix(rows as i32, cols as i32);
    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    result
}

pub fn matrix_subtract(a: &TMatrix, b: &TMatrix) -> TMatrix {
    let rows = a.len();
    if rows == 0 {
        return vec![];
    }
    let cols = a[0].len();
    let mut result = create_matrix(rows as i32, cols as i32);
    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    result
}

pub fn matrix_scale(a: &TMatrix, s: f32) -> TMatrix {
    let rows = a.len();
    if rows == 0 {
        return vec![];
    }
    let cols = a[0].len();
    let mut result = create_matrix(rows as i32, cols as i32);
    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = a[i][j] * s;
        }
    }
    result
}

pub fn matrix_transpose(a: &TMatrix) -> TMatrix {
    if a.is_empty() {
        return vec![];
    }
    let rows = a.len();
    let cols = a[0].len();
    let mut result = create_matrix(cols as i32, rows as i32);
    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = a[i][j];
        }
    }
    result
}

pub fn matrix_normalize(a: &TMatrix) -> TMatrix {
    if a.is_empty() {
        return vec![];
    }
    let rows = a.len();
    let cols = a[0].len();
    let mut result = create_matrix(rows as i32, cols as i32);
    for i in 0..rows {
        let mut norm = 0.0f32;
        for j in 0..cols {
            norm += a[i][j] * a[i][j];
        }
        norm = norm.sqrt();
        if norm > 1e-12 {
            for j in 0..cols {
                result[i][j] = a[i][j] / norm;
            }
        } else {
            for j in 0..cols {
                result[i][j] = a[i][j];
            }
        }
    }
    result
}

pub fn matrix_element_mul(a: &TMatrix, b: &TMatrix) -> TMatrix {
    let rows = a.len();
    if rows == 0 {
        return vec![];
    }
    let cols = a[0].len();
    let mut result = create_matrix(rows as i32, cols as i32);
    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = a[i][j] * b[i][j];
        }
    }
    result
}

pub fn matrix_add_in_place(a: &mut TMatrix, b: &TMatrix) {
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            a[i][j] += b[i][j];
        }
    }
}

pub fn matrix_scale_in_place(a: &mut TMatrix, s: f32) {
    for row in a.iter_mut() {
        for val in row.iter_mut() {
            *val *= s;
        }
    }
}

pub fn matrix_clip_in_place(a: &mut TMatrix, lo: f32, hi: f32) {
    for row in a.iter_mut() {
        for val in row.iter_mut() {
            if *val < lo {
                *val = lo;
            } else if *val > hi {
                *val = hi;
            }
        }
    }
}

pub fn safe_get(m: &TMatrix, r: i32, c: i32, def: f32) -> f32 {
    if r >= 0 && (r as usize) < m.len() && c >= 0 && (c as usize) < m[r as usize].len() {
        m[r as usize][c as usize]
    } else {
        def
    }
}

pub fn safe_set(m: &mut TMatrix, r: i32, c: i32, val: f32) {
    if r >= 0 && (r as usize) < m.len() && c >= 0 && (c as usize) < m[r as usize].len() {
        m[r as usize][c as usize] = val;
    }
}

/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Activation functions — port of GF_Op_ activation functions.
 */

use crate::matrix::create_matrix;
use crate::types::{ActivationType, TMatrix};

pub fn matrix_relu(a: &TMatrix) -> TMatrix {
    let rows = a.len();
    if rows == 0 {
        return vec![];
    }
    let cols = a[0].len();
    let mut result = create_matrix(rows as i32, cols as i32);
    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = if a[i][j] > 0.0 { a[i][j] } else { 0.0 };
        }
    }
    result
}

pub fn matrix_leaky_relu(a: &TMatrix, alpha: f32) -> TMatrix {
    let rows = a.len();
    if rows == 0 {
        return vec![];
    }
    let cols = a[0].len();
    let mut result = create_matrix(rows as i32, cols as i32);
    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = if a[i][j] > 0.0 {
                a[i][j]
            } else {
                alpha * a[i][j]
            };
        }
    }
    result
}

pub fn matrix_sigmoid(a: &TMatrix) -> TMatrix {
    let rows = a.len();
    if rows == 0 {
        return vec![];
    }
    let cols = a[0].len();
    let mut result = create_matrix(rows as i32, cols as i32);
    for i in 0..rows {
        for j in 0..cols {
            let v = a[i][j].clamp(-88.0, 88.0);
            result[i][j] = 1.0 / (1.0 + (-v).exp());
        }
    }
    result
}

pub fn matrix_tanh(a: &TMatrix) -> TMatrix {
    let rows = a.len();
    if rows == 0 {
        return vec![];
    }
    let cols = a[0].len();
    let mut result = create_matrix(rows as i32, cols as i32);
    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = a[i][j].tanh();
        }
    }
    result
}

pub fn matrix_softmax(a: &TMatrix) -> TMatrix {
    let rows = a.len();
    if rows == 0 {
        return vec![];
    }
    let cols = a[0].len();
    let mut result = create_matrix(rows as i32, cols as i32);
    for i in 0..rows {
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..cols {
            if a[i][j] > max_val {
                max_val = a[i][j];
            }
        }
        let mut sum = 0.0f32;
        for j in 0..cols {
            let v = (a[i][j] - max_val).exp();
            result[i][j] = v;
            sum += v;
        }
        if sum > 0.0 {
            for j in 0..cols {
                result[i][j] /= sum;
            }
        }
    }
    result
}

pub fn apply_activation(a: &TMatrix, act: ActivationType) -> TMatrix {
    match act {
        ActivationType::ReLU => matrix_relu(a),
        ActivationType::Sigmoid => matrix_sigmoid(a),
        ActivationType::Tanh => matrix_tanh(a),
        ActivationType::LeakyReLU => matrix_leaky_relu(a, 0.01),
        ActivationType::None => a.clone(),
    }
}

pub fn activation_backward(grad_out: &TMatrix, pre_act: &TMatrix, act: ActivationType) -> TMatrix {
    let rows = grad_out.len();
    if rows == 0 {
        return vec![];
    }
    let cols = grad_out[0].len();
    let mut result = create_matrix(rows as i32, cols as i32);
    match act {
        ActivationType::ReLU => {
            for i in 0..rows {
                for j in 0..cols {
                    result[i][j] = if pre_act[i][j] > 0.0 {
                        grad_out[i][j]
                    } else {
                        0.0
                    };
                }
            }
        }
        ActivationType::LeakyReLU => {
            let alpha = 0.01f32;
            for i in 0..rows {
                for j in 0..cols {
                    result[i][j] = if pre_act[i][j] > 0.0 {
                        grad_out[i][j]
                    } else {
                        alpha * grad_out[i][j]
                    };
                }
            }
        }
        ActivationType::Sigmoid => {
            for i in 0..rows {
                for j in 0..cols {
                    let v = pre_act[i][j].clamp(-88.0, 88.0);
                    let s = 1.0 / (1.0 + (-v).exp());
                    result[i][j] = grad_out[i][j] * s * (1.0 - s);
                }
            }
        }
        ActivationType::Tanh => {
            for i in 0..rows {
                for j in 0..cols {
                    let t = pre_act[i][j].tanh();
                    result[i][j] = grad_out[i][j] * (1.0 - t * t);
                }
            }
        }
        ActivationType::None => {
            for i in 0..rows {
                for j in 0..cols {
                    result[i][j] = grad_out[i][j];
                }
            }
        }
    }
    result
}

/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Normalization operations — port of GF_Op_ normalization functions.
 */

use crate::matrix::create_matrix;
use crate::random::random_gaussian;
use crate::types::{Layer, TMatrix};

pub fn batch_norm_forward(inp: &TMatrix, layer: &mut Layer) -> TMatrix {
    let batch = inp.len();
    if batch == 0 {
        return vec![];
    }
    let features = inp[0].len();
    layer.layer_input = inp.clone();

    let mut result = create_matrix(batch as i32, features as i32);
    let mut mean = vec![0.0f32; features];
    let mut var = vec![0.0f32; features];

    if layer.is_training {
        // Compute batch mean
        for j in 0..features {
            let mut s = 0.0f32;
            for i in 0..batch {
                s += inp[i][j];
            }
            mean[j] = s / batch as f32;
        }
        // Compute batch variance
        for j in 0..features {
            let mut s = 0.0f32;
            for i in 0..batch {
                let d = inp[i][j] - mean[j];
                s += d * d;
            }
            var[j] = s / batch as f32;
        }
        // Update running stats
        if layer.running_mean.len() == features {
            let mom = layer.bn_momentum;
            for j in 0..features {
                layer.running_mean[j] = (1.0 - mom) * layer.running_mean[j] + mom * mean[j];
                layer.running_var[j] = (1.0 - mom) * layer.running_var[j] + mom * var[j];
            }
        }
    } else {
        mean = layer.running_mean.clone();
        var = layer.running_var.clone();
    }

    let eps = layer.bn_epsilon;
    for i in 0..batch {
        for j in 0..features {
            let norm = (inp[i][j] - mean[j]) / (var[j] + eps).sqrt();
            result[i][j] = if j < layer.bn_gamma.len() {
                layer.bn_gamma[j] * norm + layer.bn_beta[j]
            } else {
                norm
            };
        }
    }

    layer.cached_mean = mean;
    layer.cached_var = var;
    layer.cached_normalized = result.clone();
    layer.layer_output = result.clone();
    result
}

pub fn batch_norm_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    let batch = grad_out.len();
    if batch == 0 {
        return vec![];
    }
    let features = grad_out[0].len();
    let eps = layer.bn_epsilon;

    let mut grad_input = create_matrix(batch as i32, features as i32);

    // Gamma/beta gradients
    if layer.bn_gamma_grad.is_empty() {
        layer.bn_gamma_grad = vec![0.0; features];
    }
    if layer.bn_beta_grad.is_empty() {
        layer.bn_beta_grad = vec![0.0; features];
    }

    for j in 0..features {
        let mean = if j < layer.cached_mean.len() {
            layer.cached_mean[j]
        } else {
            0.0
        };
        let var = if j < layer.cached_var.len() {
            layer.cached_var[j]
        } else {
            1.0
        };
        let std = (var + eps).sqrt();
        let gamma = if j < layer.bn_gamma.len() {
            layer.bn_gamma[j]
        } else {
            1.0
        };

        let mut d_gamma = 0.0f32;
        let mut d_beta = 0.0f32;
        let mut d_norm_sum = 0.0f32;
        let mut d_norm_x_sum = 0.0f32;

        for i in 0..batch {
            let x_norm = (layer.layer_input[i][j] - mean) / std;
            d_gamma += grad_out[i][j] * x_norm;
            d_beta += grad_out[i][j];
            let d_norm = grad_out[i][j] * gamma;
            d_norm_sum += d_norm;
            d_norm_x_sum += d_norm * (layer.layer_input[i][j] - mean);
        }

        layer.bn_gamma_grad[j] = d_gamma;
        layer.bn_beta_grad[j] = d_beta;

        for i in 0..batch {
            let d_norm = grad_out[i][j] * gamma;
            grad_input[i][j] = (1.0 / (batch as f32))
                * (1.0 / std)
                * (batch as f32 * d_norm
                    - d_norm_sum
                    - (layer.layer_input[i][j] - mean) * d_norm_x_sum / (var + eps));
        }
    }

    grad_input
}

pub fn layer_norm_forward(inp: &TMatrix, layer: &mut Layer) -> TMatrix {
    let batch = inp.len();
    if batch == 0 {
        return vec![];
    }
    let features = inp[0].len();
    layer.layer_input = inp.clone();

    let eps = layer.bn_epsilon;
    let mut result = create_matrix(batch as i32, features as i32);
    let mut means = vec![0.0f32; batch];
    let mut vars = vec![0.0f32; batch];

    for i in 0..batch {
        let mut s = 0.0f32;
        for j in 0..features {
            s += inp[i][j];
        }
        means[i] = s / features as f32;

        let mut v = 0.0f32;
        for j in 0..features {
            let d = inp[i][j] - means[i];
            v += d * d;
        }
        vars[i] = v / features as f32;

        for j in 0..features {
            let norm = (inp[i][j] - means[i]) / (vars[i] + eps).sqrt();
            result[i][j] = if j < layer.bn_gamma.len() {
                layer.bn_gamma[j] * norm + layer.bn_beta[j]
            } else {
                norm
            };
        }
    }

    layer.cached_mean = means;
    layer.cached_var = vars;
    layer.cached_normalized = result.clone();
    layer.layer_output = result.clone();
    result
}

pub fn layer_norm_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    let batch = grad_out.len();
    if batch == 0 {
        return vec![];
    }
    let features = grad_out[0].len();
    let eps = layer.bn_epsilon;

    let mut grad_input = create_matrix(batch as i32, features as i32);

    if layer.bn_gamma_grad.is_empty() {
        layer.bn_gamma_grad = vec![0.0; features];
    }
    if layer.bn_beta_grad.is_empty() {
        layer.bn_beta_grad = vec![0.0; features];
    }

    for i in 0..batch {
        let mean = layer.cached_mean[i];
        let var = layer.cached_var[i];
        let std = (var + eps).sqrt();

        let mut d_norm_sum = 0.0f32;
        let mut d_norm_x_sum = 0.0f32;

        for j in 0..features {
            let gamma = if j < layer.bn_gamma.len() {
                layer.bn_gamma[j]
            } else {
                1.0
            };
            let d_norm = grad_out[i][j] * gamma;
            d_norm_sum += d_norm;
            d_norm_x_sum += d_norm * (layer.layer_input[i][j] - mean);

            let x_norm = (layer.layer_input[i][j] - mean) / std;
            layer.bn_gamma_grad[j] += grad_out[i][j] * x_norm;
            layer.bn_beta_grad[j] += grad_out[i][j];
        }

        for j in 0..features {
            let gamma = if j < layer.bn_gamma.len() {
                layer.bn_gamma[j]
            } else {
                1.0
            };
            let d_norm = grad_out[i][j] * gamma;
            grad_input[i][j] = (1.0 / (features as f32))
                * (1.0 / std)
                * (features as f32 * d_norm
                    - d_norm_sum
                    - (layer.layer_input[i][j] - mean) * d_norm_x_sum / (var + eps));
        }
    }

    grad_input
}

pub fn spectral_normalize(layer: &mut Layer) -> TMatrix {
    if layer.weights.is_empty() || layer.spectral_u.is_empty() || layer.spectral_v.is_empty() {
        return layer.weights.clone();
    }

    let rows = layer.weights.len();
    let cols = layer.weights[0].len();

    // Power iteration
    // v = W^T u / ||W^T u||
    let mut v_new = vec![0.0f32; cols];
    for j in 0..cols {
        let mut s = 0.0f32;
        for i in 0..rows {
            if i < layer.spectral_u.len() {
                s += layer.weights[i][j] * layer.spectral_u[i];
            }
        }
        v_new[j] = s;
    }
    let mut norm = 0.0f32;
    for j in 0..cols {
        norm += v_new[j] * v_new[j];
    }
    norm = norm.sqrt().max(1e-12);
    for j in 0..cols {
        v_new[j] /= norm;
    }

    // u = W v / ||W v||
    let mut u_new = vec![0.0f32; rows];
    for i in 0..rows {
        let mut s = 0.0f32;
        for j in 0..cols {
            s += layer.weights[i][j] * v_new[j];
        }
        u_new[i] = s;
    }
    norm = 0.0;
    for i in 0..rows {
        norm += u_new[i] * u_new[i];
    }
    norm = norm.sqrt().max(1e-12);
    for i in 0..rows {
        u_new[i] /= norm;
    }

    // sigma = u^T W v
    let mut sigma = 0.0f32;
    for i in 0..rows {
        let mut s = 0.0f32;
        for j in 0..cols {
            s += layer.weights[i][j] * v_new[j];
        }
        sigma += u_new[i] * s;
    }

    layer.spectral_u = u_new;
    layer.spectral_v = v_new;
    layer.spectral_sigma = sigma;

    // Return normalized weights
    let mut result = create_matrix(rows as i32, cols as i32);
    if sigma.abs() > 1e-12 {
        for i in 0..rows {
            for j in 0..cols {
                result[i][j] = layer.weights[i][j] / sigma;
            }
        }
    } else {
        result = layer.weights.clone();
    }
    result
}

pub fn init_spectral_vectors(layer: &mut Layer) {
    let rows = layer.weights.len();
    let cols = if rows > 0 { layer.weights[0].len() } else { 0 };
    layer.spectral_u = (0..rows).map(|_| random_gaussian()).collect();
    layer.spectral_v = (0..cols).map(|_| random_gaussian()).collect();
}

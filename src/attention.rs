/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Self-attention — port of GF_Op_ attention functions.
 */

use crate::matrix::{create_matrix, matrix_multiply, matrix_transpose};
use crate::types::{Layer, TMatrix};

pub fn self_attention_forward(inp: &TMatrix, layer: &mut Layer) -> TMatrix {
    let seq_len = inp.len();
    if seq_len == 0 {
        return vec![];
    }
    let d_model = inp[0].len();
    layer.layer_input = inp.clone();

    // Q = inp * Wq, K = inp * Wk, V = inp * Wv
    let q = matrix_multiply(inp, &layer.wq);
    let k = matrix_multiply(inp, &layer.wk);
    let v = matrix_multiply(inp, &layer.wv);

    layer.cached_q = q.clone();
    layer.cached_k = k.clone();
    layer.cached_v = v.clone();

    // Scaled dot-product attention: scores = Q * K^T / sqrt(d_k)
    let kt = matrix_transpose(&k);
    let mut scores = matrix_multiply(&q, &kt);

    let d_k = (d_model as f32).sqrt();
    for i in 0..scores.len() {
        for j in 0..scores[i].len() {
            scores[i][j] /= d_k;
        }
    }

    // Softmax over each row
    for i in 0..scores.len() {
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..scores[i].len() {
            if scores[i][j] > max_val {
                max_val = scores[i][j];
            }
        }
        let mut sum = 0.0f32;
        for j in 0..scores[i].len() {
            scores[i][j] = (scores[i][j] - max_val).exp();
            sum += scores[i][j];
        }
        if sum > 0.0 {
            for j in 0..scores[i].len() {
                scores[i][j] /= sum;
            }
        }
    }

    layer.cached_scores = scores.clone();

    // attended = scores * V
    let attended = matrix_multiply(&scores, &v);
    layer.cached_attended = attended.clone();

    // output = attended * Wo
    let output = matrix_multiply(&attended, &layer.wo);
    layer.layer_output = output.clone();
    layer.pre_activation = output.clone();
    output
}

pub fn self_attention_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    let seq_len = grad_out.len();
    if seq_len == 0 {
        return vec![];
    }

    // grad through Wo
    let wo_t = matrix_transpose(&layer.wo);
    let d_attended = matrix_multiply(grad_out, &wo_t);

    // Wo gradient: attended^T * grad_out
    let att_t = matrix_transpose(&layer.cached_attended);
    layer.wo_grad = matrix_multiply(&att_t, grad_out);

    // grad through scores * V
    let vt = matrix_transpose(&layer.cached_v);
    let d_scores = matrix_multiply(&d_attended, &vt);

    let scores_t = matrix_transpose(&layer.cached_scores);
    let d_v = matrix_multiply(&scores_t, &d_attended);

    // Softmax backward on d_scores
    let mut d_scores_pre = create_matrix(seq_len as i32, seq_len as i32);
    for i in 0..seq_len {
        for j in 0..seq_len {
            let s = layer.cached_scores[i][j];
            d_scores_pre[i][j] = s * (d_scores[i][j] - {
                let mut dot = 0.0f32;
                for k in 0..seq_len {
                    dot += d_scores[i][k] * layer.cached_scores[i][k];
                }
                dot
            });
        }
    }

    let d_model = layer.layer_input[0].len();
    let d_k = (d_model as f32).sqrt();
    for i in 0..d_scores_pre.len() {
        for j in 0..d_scores_pre[i].len() {
            d_scores_pre[i][j] /= d_k;
        }
    }

    // grad Q, K from scores = Q * K^T
    let d_q = matrix_multiply(&d_scores_pre, &layer.cached_k);
    let d_scores_t = matrix_transpose(&d_scores_pre);
    let d_k_mat = matrix_multiply(&d_scores_t, &layer.cached_q);

    // Gradients for Wq, Wk, Wv
    let inp_t = matrix_transpose(&layer.layer_input);
    layer.wq_grad = matrix_multiply(&inp_t, &d_q);
    layer.wk_grad = matrix_multiply(&inp_t, &d_k_mat);
    layer.wv_grad = matrix_multiply(&inp_t, &d_v);

    // Gradient for input
    let wq_t = matrix_transpose(&layer.wq);
    let wk_t = matrix_transpose(&layer.wk);
    let wv_t = matrix_transpose(&layer.wv);

    let gi_q = matrix_multiply(&d_q, &wq_t);
    let gi_k = matrix_multiply(&d_k_mat, &wk_t);
    let gi_v = matrix_multiply(&d_v, &wv_t);

    let mut grad_input = create_matrix(seq_len as i32, d_model as i32);
    for i in 0..seq_len {
        for j in 0..d_model {
            grad_input[i][j] = gi_q[i][j] + gi_k[i][j] + gi_v[i][j];
        }
    }

    grad_input
}

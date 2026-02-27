/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Layer creation and dispatch — port of GF_Op_ layer functions.
 */

use crate::activations::*;
use crate::attention::*;
use crate::convolution::*;
use crate::matrix::create_matrix;
use crate::normalization::*;
use crate::random::random_gaussian;
use crate::types::*;

pub fn create_dense_layer(in_sz: i32, out_sz: i32, act: ActivationType) -> Layer {
    let mut layer = Layer::default();
    layer.layer_type = LayerType::Dense;
    layer.activation = act;
    layer.input_size = in_sz;
    layer.output_size = out_sz;

    // Xavier initialization
    let scale = (2.0 / (in_sz + out_sz) as f32).sqrt();
    layer.weights = (0..in_sz as usize)
        .map(|_| {
            (0..out_sz as usize)
                .map(|_| random_gaussian() * scale)
                .collect()
        })
        .collect();
    layer.bias = vec![0.0; out_sz as usize];
    layer
}

pub fn create_conv2d_layer(
    in_ch: i32,
    out_ch: i32,
    k_sz: i32,
    st: i32,
    pad: i32,
    w: i32,
    h: i32,
    act: ActivationType,
) -> Layer {
    let mut layer = Layer::default();
    layer.layer_type = LayerType::Conv2D;
    layer.activation = act;
    layer.in_channels = in_ch;
    layer.out_channels = out_ch;
    layer.kernel_size = k_sz;
    layer.stride = st.max(1);
    layer.padding = pad;
    layer.in_width = w;
    layer.in_height = h;
    layer.input_size = in_ch * w * h;

    let out_w = (w + 2 * pad - k_sz) / st.max(1) + 1;
    let out_h = (h + 2 * pad - k_sz) / st.max(1) + 1;
    layer.out_width = out_w;
    layer.out_height = out_h;
    layer.output_size = out_ch * out_w * out_h;

    let fan_in = (in_ch * k_sz * k_sz) as f32;
    let scale = (2.0 / fan_in).sqrt();

    layer.kernels = (0..out_ch as usize)
        .map(|_| {
            (0..(in_ch * k_sz * k_sz) as usize)
                .map(|_| vec![random_gaussian() * scale])
                .collect()
        })
        .collect();
    layer.bias = vec![0.0; out_ch as usize];
    layer
}

pub fn create_deconv2d_layer(
    in_ch: i32,
    out_ch: i32,
    k_sz: i32,
    st: i32,
    pad: i32,
    w: i32,
    h: i32,
    act: ActivationType,
) -> Layer {
    let mut layer = Layer::default();
    layer.layer_type = LayerType::Deconv2D;
    layer.activation = act;
    layer.in_channels = in_ch;
    layer.out_channels = out_ch;
    layer.kernel_size = k_sz;
    layer.stride = st.max(1);
    layer.padding = pad;
    layer.in_width = w;
    layer.in_height = h;
    layer.input_size = in_ch * w * h;

    let out_w = (w - 1) * st.max(1) + k_sz - 2 * pad;
    let out_h = (h - 1) * st.max(1) + k_sz - 2 * pad;
    layer.out_width = out_w;
    layer.out_height = out_h;
    layer.output_size = out_ch * out_w * out_h;

    let fan_in = (in_ch * k_sz * k_sz) as f32;
    let scale = (2.0 / fan_in).sqrt();

    layer.kernels = (0..out_ch as usize)
        .map(|_| {
            (0..(in_ch * k_sz * k_sz) as usize)
                .map(|_| vec![random_gaussian() * scale])
                .collect()
        })
        .collect();
    layer.bias = vec![0.0; out_ch as usize];
    layer
}

pub fn create_conv1d_layer(
    in_ch: i32,
    out_ch: i32,
    k_sz: i32,
    st: i32,
    pad: i32,
    in_len: i32,
    act: ActivationType,
) -> Layer {
    let mut layer = Layer::default();
    layer.layer_type = LayerType::Conv1D;
    layer.activation = act;
    layer.in_channels = in_ch;
    layer.out_channels = out_ch;
    layer.kernel_size = k_sz;
    layer.stride = st.max(1);
    layer.padding = pad;
    layer.in_width = in_len;
    layer.input_size = in_ch * in_len;

    let out_len = (in_len + 2 * pad - k_sz) / st.max(1) + 1;
    layer.out_width = out_len;
    layer.output_size = out_ch * out_len;

    let fan_in = (in_ch * k_sz) as f32;
    let scale = (2.0 / fan_in).sqrt();

    layer.kernels = (0..out_ch as usize)
        .map(|_| {
            (0..(in_ch * k_sz) as usize)
                .map(|_| vec![random_gaussian() * scale])
                .collect()
        })
        .collect();
    layer.bias = vec![0.0; out_ch as usize];
    layer
}

pub fn create_batch_norm_layer(features: i32) -> Layer {
    let mut layer = Layer::default();
    layer.layer_type = LayerType::BatchNorm;
    layer.input_size = features;
    layer.output_size = features;
    layer.bn_gamma = vec![1.0; features as usize];
    layer.bn_beta = vec![0.0; features as usize];
    layer.running_mean = vec![0.0; features as usize];
    layer.running_var = vec![1.0; features as usize];
    layer.bn_epsilon = DEFAULT_BN_EPS;
    layer.bn_momentum = DEFAULT_BN_MOM;
    layer
}

pub fn create_layer_norm_layer(features: i32) -> Layer {
    let mut layer = Layer::default();
    layer.layer_type = LayerType::LayerNorm;
    layer.input_size = features;
    layer.output_size = features;
    layer.bn_gamma = vec![1.0; features as usize];
    layer.bn_beta = vec![0.0; features as usize];
    layer.bn_epsilon = DEFAULT_BN_EPS;
    layer.bn_momentum = DEFAULT_BN_MOM;
    layer
}

pub fn create_attention_layer(d_model: i32, n_heads: i32) -> Layer {
    let mut layer = Layer::default();
    layer.layer_type = LayerType::Attention;
    layer.input_size = d_model;
    layer.output_size = d_model;
    layer.num_heads = n_heads;
    layer.head_dim = d_model / n_heads;

    let scale = (2.0 / (d_model + d_model) as f32).sqrt();
    let init_mat = |rows: usize, cols: usize| -> TMatrix {
        (0..rows)
            .map(|_| (0..cols).map(|_| random_gaussian() * scale).collect())
            .collect()
    };

    layer.wq = init_mat(d_model as usize, d_model as usize);
    layer.wk = init_mat(d_model as usize, d_model as usize);
    layer.wv = init_mat(d_model as usize, d_model as usize);
    layer.wo = init_mat(d_model as usize, d_model as usize);
    layer
}

pub fn layer_forward(layer: &mut Layer, inp: &TMatrix) -> TMatrix {
    match layer.layer_type {
        LayerType::Dense => dense_forward(layer, inp),
        LayerType::Conv2D => {
            let out = conv2d_forward(inp, layer);
            apply_activation(&out, layer.activation)
        }
        LayerType::Deconv2D => {
            let out = deconv2d_forward(inp, layer);
            apply_activation(&out, layer.activation)
        }
        LayerType::Conv1D => {
            let out = conv1d_forward(inp, layer);
            apply_activation(&out, layer.activation)
        }
        LayerType::BatchNorm => batch_norm_forward(inp, layer),
        LayerType::LayerNorm => layer_norm_forward(inp, layer),
        LayerType::SpectralNorm => {
            spectral_normalize(layer);
            inp.clone()
        }
        LayerType::Attention => self_attention_forward(inp, layer),
    }
}

fn dense_forward(layer: &mut Layer, inp: &TMatrix) -> TMatrix {
    let batch = inp.len();
    if batch == 0 {
        return vec![];
    }
    let in_sz = layer.input_size as usize;
    let out_sz = layer.output_size as usize;
    layer.layer_input = inp.clone();

    let mut pre_act = create_matrix(batch as i32, out_sz as i32);
    for b in 0..batch {
        for j in 0..out_sz {
            let mut sum = layer.bias[j];
            for i in 0..in_sz {
                if i < inp[b].len() && i < layer.weights.len() && j < layer.weights[i].len() {
                    sum += inp[b][i] * layer.weights[i][j];
                }
            }
            pre_act[b][j] = sum;
        }
    }
    layer.pre_activation = pre_act.clone();
    let output = apply_activation(&pre_act, layer.activation);
    layer.layer_output = output.clone();
    output
}

pub fn layer_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    match layer.layer_type {
        LayerType::Dense => dense_backward(layer, grad_out),
        LayerType::Conv2D => {
            let grad_pre = activation_backward(grad_out, &layer.pre_activation, layer.activation);
            conv2d_backward(layer, &grad_pre)
        }
        LayerType::Deconv2D => {
            let grad_pre = activation_backward(grad_out, &layer.pre_activation, layer.activation);
            deconv2d_backward(layer, &grad_pre)
        }
        LayerType::Conv1D => {
            let grad_pre = activation_backward(grad_out, &layer.pre_activation, layer.activation);
            conv1d_backward(layer, &grad_pre)
        }
        LayerType::BatchNorm => batch_norm_backward(layer, grad_out),
        LayerType::LayerNorm => layer_norm_backward(layer, grad_out),
        LayerType::SpectralNorm => grad_out.clone(),
        LayerType::Attention => self_attention_backward(layer, grad_out),
    }
}

fn dense_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    let batch = grad_out.len();
    if batch == 0 {
        return vec![];
    }
    let in_sz = layer.input_size as usize;
    let out_sz = layer.output_size as usize;

    // Apply activation backward
    let grad_pre = activation_backward(grad_out, &layer.pre_activation, layer.activation);

    // Weight gradient
    layer.weight_grad = create_matrix(in_sz as i32, out_sz as i32);
    layer.bias_grad = vec![0.0; out_sz];

    for b in 0..batch {
        for i in 0..in_sz {
            for j in 0..out_sz {
                if i < layer.layer_input[b].len() {
                    layer.weight_grad[i][j] += layer.layer_input[b][i] * grad_pre[b][j];
                }
            }
        }
        for j in 0..out_sz {
            layer.bias_grad[j] += grad_pre[b][j];
        }
    }

    // Input gradient
    let mut grad_input = create_matrix(batch as i32, in_sz as i32);
    for b in 0..batch {
        for i in 0..in_sz {
            let mut sum = 0.0f32;
            for j in 0..out_sz {
                if i < layer.weights.len() && j < layer.weights[i].len() {
                    sum += grad_pre[b][j] * layer.weights[i][j];
                }
            }
            grad_input[b][i] = sum;
        }
    }

    grad_input
}

pub fn init_layer_optimizer(layer: &mut Layer, opt: Optimizer) {
    let in_sz = layer.input_size as usize;
    let out_sz = layer.output_size as usize;

    match opt {
        Optimizer::Adam => {
            if layer.layer_type == LayerType::Dense {
                layer.m_weight = create_matrix(in_sz as i32, out_sz as i32);
                layer.v_weight = create_matrix(in_sz as i32, out_sz as i32);
                layer.m_bias = vec![0.0; out_sz];
                layer.v_bias = vec![0.0; out_sz];
            }
            if layer.layer_type == LayerType::Attention {
                let d = layer.input_size as usize;
                layer.m_wq = create_matrix(d as i32, d as i32);
                layer.v_wq = create_matrix(d as i32, d as i32);
                layer.m_wk = create_matrix(d as i32, d as i32);
                layer.v_wk = create_matrix(d as i32, d as i32);
                layer.m_wv = create_matrix(d as i32, d as i32);
                layer.v_wv = create_matrix(d as i32, d as i32);
                layer.m_wo = create_matrix(d as i32, d as i32);
                layer.v_wo = create_matrix(d as i32, d as i32);
            }
            if layer.layer_type == LayerType::BatchNorm
                || layer.layer_type == LayerType::LayerNorm
            {
                let f = layer.input_size as usize;
                layer.m_bn_gamma = vec![0.0; f];
                layer.v_bn_gamma = vec![0.0; f];
                layer.m_bn_beta = vec![0.0; f];
                layer.v_bn_beta = vec![0.0; f];
            }
        }
        Optimizer::SGD => {}
        Optimizer::RMSProp => {
            if layer.layer_type == LayerType::Dense {
                layer.rms_weight = create_matrix(in_sz as i32, out_sz as i32);
                layer.rms_bias = vec![0.0; out_sz];
            }
        }
    }
}

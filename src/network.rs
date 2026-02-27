/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Network operations — port of GF_Gen_, GF_Disc_, and network functions.
 */

use crate::layer::*;
use crate::random::generate_noise;
use crate::types::*;

pub fn create_network(
    sizes: &[i32],
    act: ActivationType,
    opt: Optimizer,
    lr: f32,
) -> Network {
    let mut net = Network::default();
    net.optimizer = opt;
    net.learning_rate = lr;
    net.beta1 = 0.9;
    net.beta2 = 0.999;
    net.epsilon = 1e-8;
    net.progressive_alpha = 1.0;
    net.is_training = true;

    let num_layers = sizes.len() - 1;
    for i in 0..num_layers {
        let layer_act = if i == num_layers - 1 {
            ActivationType::Sigmoid
        } else {
            act
        };
        let mut layer = create_dense_layer(sizes[i], sizes[i + 1], layer_act);
        init_layer_optimizer(&mut layer, opt);
        net.layers.push(layer);
    }
    net.layer_count = num_layers as i32;
    net
}

pub fn create_conv_generator(
    noise_dim: i32,
    cond_sz: i32,
    base_ch: i32,
    act: ActivationType,
    opt: Optimizer,
    lr: f32,
) -> Network {
    let mut net = Network::default();
    net.optimizer = opt;
    net.learning_rate = lr;
    net.beta1 = 0.9;
    net.beta2 = 0.999;
    net.epsilon = 1e-8;
    net.progressive_alpha = 1.0;
    net.is_training = true;

    // Dense: noise_dim+cond_sz -> base_ch*4 * 4*4
    let total_in = noise_dim + cond_sz;
    let dense_out = base_ch * 4 * 4 * 4;
    let mut l0 = create_dense_layer(total_in, dense_out, act);
    init_layer_optimizer(&mut l0, opt);
    net.layers.push(l0);

    // BatchNorm
    let mut bn1 = create_batch_norm_layer(dense_out);
    init_layer_optimizer(&mut bn1, opt);
    net.layers.push(bn1);

    // Deconv layers
    let mut l2 = create_deconv2d_layer(base_ch * 4, base_ch * 2, 4, 2, 1, 4, 4, act);
    init_layer_optimizer(&mut l2, opt);
    net.layers.push(l2);

    let mut bn2 = create_batch_norm_layer(base_ch * 2 * 8 * 8);
    init_layer_optimizer(&mut bn2, opt);
    net.layers.push(bn2);

    let mut l4 = create_deconv2d_layer(base_ch * 2, base_ch, 4, 2, 1, 8, 8, act);
    init_layer_optimizer(&mut l4, opt);
    net.layers.push(l4);

    let mut bn3 = create_batch_norm_layer(base_ch * 16 * 16);
    init_layer_optimizer(&mut bn3, opt);
    net.layers.push(bn3);

    // Final deconv to 1 channel
    let mut l6 = create_deconv2d_layer(base_ch, 1, 4, 2, 1, 16, 16, ActivationType::Tanh);
    init_layer_optimizer(&mut l6, opt);
    net.layers.push(l6);

    net.layer_count = net.layers.len() as i32;
    net
}

pub fn create_conv_discriminator(
    in_ch: i32,
    in_w: i32,
    in_h: i32,
    cond_sz: i32,
    base_ch: i32,
    act: ActivationType,
    opt: Optimizer,
    lr: f32,
) -> Network {
    let mut net = Network::default();
    net.optimizer = opt;
    net.learning_rate = lr;
    net.beta1 = 0.9;
    net.beta2 = 0.999;
    net.epsilon = 1e-8;
    net.progressive_alpha = 1.0;
    net.is_training = true;

    let mut l0 = create_conv2d_layer(in_ch, base_ch, 4, 2, 1, in_w, in_h, act);
    init_layer_optimizer(&mut l0, opt);
    net.layers.push(l0);

    let w1 = in_w / 2;
    let h1 = in_h / 2;
    let mut l1 = create_conv2d_layer(base_ch, base_ch * 2, 4, 2, 1, w1, h1, act);
    init_layer_optimizer(&mut l1, opt);
    net.layers.push(l1);

    let w2 = w1 / 2;
    let h2 = h1 / 2;
    let mut l2 = create_conv2d_layer(base_ch * 2, base_ch * 4, 4, 2, 1, w2, h2, act);
    init_layer_optimizer(&mut l2, opt);
    net.layers.push(l2);

    let _w3 = w2 / 2;
    let _h3 = h2 / 2;

    // Flatten + Dense
    let flat_sz = base_ch * 4 * (w2 / 2) * (h2 / 2) + cond_sz;
    let mut l3 = create_dense_layer(flat_sz, 1, ActivationType::Sigmoid);
    init_layer_optimizer(&mut l3, opt);

    // BatchNorm after second conv
    let mut bn = create_batch_norm_layer(base_ch * 2 * w2 * h2);
    init_layer_optimizer(&mut bn, opt);
    net.layers.push(bn);

    net.layers.push(l3);

    net.layer_count = net.layers.len() as i32;
    net
}

pub fn network_forward(net: &mut Network, inp: &TMatrix) -> TMatrix {
    let mut current = inp.clone();
    for i in 0..net.layer_count as usize {
        net.layers[i].is_training = net.is_training;
        current = layer_forward(&mut net.layers[i], &current);
    }
    current
}

pub fn network_backward(net: &mut Network, grad_out: &TMatrix) -> TMatrix {
    let mut current_grad = grad_out.clone();
    for i in (0..net.layer_count as usize).rev() {
        current_grad = layer_backward(&mut net.layers[i], &current_grad);
    }
    current_grad
}

pub fn network_update_weights(net: &mut Network) {
    for i in 0..net.layer_count as usize {
        update_layer_weights(
            &mut net.layers[i],
            net.optimizer,
            net.learning_rate,
            net.beta1,
            net.beta2,
            net.epsilon,
            net.weight_decay,
        );
    }
}

fn update_layer_weights(
    layer: &mut Layer,
    opt: Optimizer,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
) {
    match layer.layer_type {
        LayerType::Dense => {
            layer.adam_t += 1;
            let t = layer.adam_t;
            match opt {
                Optimizer::Adam => {
                    adam_update_matrix(
                        &mut layer.weights,
                        &layer.weight_grad,
                        &mut layer.m_weight,
                        &mut layer.v_weight,
                        t,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        wd,
                    );
                    adam_update_vector(
                        &mut layer.bias,
                        &layer.bias_grad,
                        &mut layer.m_bias,
                        &mut layer.v_bias,
                        t,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        wd,
                    );
                }
                Optimizer::SGD => {
                    sgd_update_matrix(&mut layer.weights, &layer.weight_grad, lr, wd);
                    sgd_update_vector(&mut layer.bias, &layer.bias_grad, lr, wd);
                }
                Optimizer::RMSProp => {
                    rmsprop_update_matrix(
                        &mut layer.weights,
                        &layer.weight_grad,
                        &mut layer.rms_weight,
                        lr,
                        0.9,
                        eps,
                        wd,
                    );
                    rmsprop_update_vector(
                        &mut layer.bias,
                        &layer.bias_grad,
                        &mut layer.rms_bias,
                        lr,
                        0.9,
                        eps,
                        wd,
                    );
                }
            }
        }
        LayerType::BatchNorm | LayerType::LayerNorm => {
            if !layer.bn_gamma_grad.is_empty() {
                layer.adam_t += 1;
                let t = layer.adam_t;
                adam_update_vector(
                    &mut layer.bn_gamma,
                    &layer.bn_gamma_grad,
                    &mut layer.m_bn_gamma,
                    &mut layer.v_bn_gamma,
                    t,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    0.0,
                );
                adam_update_vector(
                    &mut layer.bn_beta,
                    &layer.bn_beta_grad,
                    &mut layer.m_bn_beta,
                    &mut layer.v_bn_beta,
                    t,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    0.0,
                );
            }
        }
        LayerType::Attention => {
            layer.adam_t += 1;
            let t = layer.adam_t;
            adam_update_matrix(
                &mut layer.wq,
                &layer.wq_grad,
                &mut layer.m_wq,
                &mut layer.v_wq,
                t,
                lr,
                beta1,
                beta2,
                eps,
                wd,
            );
            adam_update_matrix(
                &mut layer.wk,
                &layer.wk_grad,
                &mut layer.m_wk,
                &mut layer.v_wk,
                t,
                lr,
                beta1,
                beta2,
                eps,
                wd,
            );
            adam_update_matrix(
                &mut layer.wv,
                &layer.wv_grad,
                &mut layer.m_wv,
                &mut layer.v_wv,
                t,
                lr,
                beta1,
                beta2,
                eps,
                wd,
            );
            adam_update_matrix(
                &mut layer.wo,
                &layer.wo_grad,
                &mut layer.m_wo,
                &mut layer.v_wo,
                t,
                lr,
                beta1,
                beta2,
                eps,
                wd,
            );
        }
        _ => {}
    }
}

fn adam_update_matrix(
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
    if p.is_empty() || g.is_empty() || m.is_empty() || v.is_empty() {
        return;
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

fn adam_update_vector(
    p: &mut TVector,
    g: &TVector,
    m: &mut TVector,
    v: &mut TVector,
    t: i32,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
) {
    if p.is_empty() || g.is_empty() || m.is_empty() || v.is_empty() {
        return;
    }
    let n = p.len().min(g.len()).min(m.len()).min(v.len());
    for i in 0..n {
        let grad = g[i] + wd * p[i];
        m[i] = b1 * m[i] + (1.0 - b1) * grad;
        v[i] = b2 * v[i] + (1.0 - b2) * grad * grad;
        let m_hat = m[i] / (1.0 - b1.powi(t));
        let v_hat = v[i] / (1.0 - b2.powi(t));
        p[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

fn sgd_update_matrix(p: &mut TMatrix, g: &TMatrix, lr: f32, wd: f32) {
    if p.is_empty() || g.is_empty() {
        return;
    }
    for i in 0..p.len().min(g.len()) {
        for j in 0..p[i].len().min(g[i].len()) {
            p[i][j] -= lr * (g[i][j] + wd * p[i][j]);
        }
    }
}

fn sgd_update_vector(p: &mut TVector, g: &TVector, lr: f32, wd: f32) {
    if p.is_empty() || g.is_empty() {
        return;
    }
    for i in 0..p.len().min(g.len()) {
        p[i] -= lr * (g[i] + wd * p[i]);
    }
}

fn rmsprop_update_matrix(
    p: &mut TMatrix,
    g: &TMatrix,
    cache: &mut TMatrix,
    lr: f32,
    decay: f32,
    eps: f32,
    wd: f32,
) {
    if p.is_empty() || g.is_empty() || cache.is_empty() {
        return;
    }
    for i in 0..p.len().min(g.len()).min(cache.len()) {
        for j in 0..p[i].len().min(g[i].len()).min(cache[i].len()) {
            let grad = g[i][j] + wd * p[i][j];
            cache[i][j] = decay * cache[i][j] + (1.0 - decay) * grad * grad;
            p[i][j] -= lr * grad / (cache[i][j].sqrt() + eps);
        }
    }
}

fn rmsprop_update_vector(
    p: &mut TVector,
    g: &TVector,
    cache: &mut TVector,
    lr: f32,
    decay: f32,
    eps: f32,
    wd: f32,
) {
    if p.is_empty() || g.is_empty() || cache.is_empty() {
        return;
    }
    for i in 0..p.len().min(g.len()).min(cache.len()) {
        let grad = g[i] + wd * p[i];
        cache[i] = decay * cache[i] + (1.0 - decay) * grad * grad;
        p[i] -= lr * grad / (cache[i].sqrt() + eps);
    }
}

pub fn add_progressive_layer(net: &mut Network, res_lvl: i32, is_gen: bool) {
    let act = if !net.layers.is_empty() {
        net.layers[0].activation
    } else {
        ActivationType::LeakyReLU
    };
    if is_gen {
        let ch = 64;
        let mut l =
            create_deconv2d_layer(ch, ch / 2, 4, 2, 1, 1 << res_lvl, 1 << res_lvl, act);
        init_layer_optimizer(&mut l, net.optimizer);
        let mut bn = create_batch_norm_layer(ch / 2 * (2 << res_lvl) * (2 << res_lvl));
        init_layer_optimizer(&mut bn, net.optimizer);
        net.layers.push(l);
        net.layers.push(bn);
    } else {
        let ch = 64;
        let mut l = create_conv2d_layer(ch / 2, ch, 4, 2, 1, 2 << res_lvl, 2 << res_lvl, act);
        init_layer_optimizer(&mut l, net.optimizer);
        let mut bn = create_batch_norm_layer(ch * (1 << res_lvl) * (1 << res_lvl));
        init_layer_optimizer(&mut bn, net.optimizer);
        net.layers.push(l);
        net.layers.push(bn);
    }
    net.layer_count = net.layers.len() as i32;
}

pub fn set_network_training(net: &mut Network, training: bool) {
    net.is_training = training;
    for layer in net.layers.iter_mut() {
        layer.is_training = training;
    }
}

pub fn get_layer_output(net: &Network, idx: i32) -> TMatrix {
    if idx >= 0 && (idx as usize) < net.layers.len() {
        net.layers[idx as usize].layer_output.clone()
    } else {
        vec![]
    }
}

pub fn deep_copy_network(src: &Network) -> Network {
    src.clone()
}

// =========================================================================
// GF_Gen_ facade functions
// =========================================================================

pub fn gf_gen_sample(gen: &mut Network, count: i32, noise_dim: i32, nt: NoiseType) -> TMatrix {
    let mut noise = vec![];
    generate_noise(&mut noise, count, noise_dim, nt);
    network_forward(gen, &noise)
}

pub fn gf_gen_sample_conditional(
    gen: &mut Network,
    count: i32,
    noise_dim: i32,
    cond_sz: i32,
    nt: NoiseType,
    cond: &TMatrix,
) -> TMatrix {
    let mut noise = vec![];
    generate_noise(&mut noise, count, noise_dim, nt);
    let mut combined = crate::matrix::create_matrix(count, noise_dim + cond_sz);
    for i in 0..count as usize {
        for j in 0..noise_dim as usize {
            combined[i][j] = noise[i][j];
        }
        for j in 0..cond_sz as usize {
            if i < cond.len() && j < cond[i].len() {
                combined[i][noise_dim as usize + j] = cond[i][j];
            }
        }
    }
    network_forward(gen, &combined)
}

pub fn gf_gen_noise(size: i32, depth: i32, nt: NoiseType) -> TMatrix {
    let mut result = vec![];
    generate_noise(&mut result, size, depth, nt);
    result
}

// =========================================================================
// GF_Disc_ specific functions
// =========================================================================

pub fn compute_gradient_penalty(
    disc: &mut Network,
    real: &TMatrix,
    fake: &TMatrix,
    lambda: f32,
) -> f32 {
    let batch = real.len();
    if batch == 0 {
        return 0.0;
    }
    let features = real[0].len();
    let mut interp = crate::matrix::create_matrix(batch as i32, features as i32);

    for i in 0..batch {
        let alpha = crate::random::random_uniform(0.0, 1.0);
        for j in 0..features {
            let r = if j < real[i].len() { real[i][j] } else { 0.0 };
            let f = if i < fake.len() && j < fake[i].len() {
                fake[i][j]
            } else {
                0.0
            };
            interp[i][j] = alpha * r + (1.0 - alpha) * f;
        }
    }

    let out = network_forward(disc, &interp);

    // Approximate gradient norm via finite differences
    let eps = 1e-4f32;
    let mut grad_norm_sum = 0.0f32;
    for i in 0..batch {
        let mut grad_norm = 0.0f32;
        for j in 0..features {
            let mut interp_plus = interp.clone();
            interp_plus[i][j] += eps;
            let out_plus = network_forward(disc, &interp_plus);
            let d = if !out_plus.is_empty()
                && !out_plus[i].is_empty()
                && !out.is_empty()
                && !out[i].is_empty()
            {
                (out_plus[i][0] - out[i][0]) / eps
            } else {
                0.0
            };
            grad_norm += d * d;
        }
        grad_norm = grad_norm.sqrt();
        grad_norm_sum += (grad_norm - 1.0) * (grad_norm - 1.0);
    }
    lambda * grad_norm_sum / batch as f32
}

pub fn feature_matching_loss(
    disc: &mut Network,
    real: &TMatrix,
    fake: &TMatrix,
    feat_layer: i32,
) -> f32 {
    network_forward(disc, real);
    let real_feat = get_layer_output(disc, feat_layer);

    network_forward(disc, fake);
    let fake_feat = get_layer_output(disc, feat_layer);

    if real_feat.is_empty() || fake_feat.is_empty() {
        return 0.0;
    }

    let mut loss = 0.0f32;
    let mut count = 0;
    for i in 0..real_feat.len().min(fake_feat.len()) {
        for j in 0..real_feat[i].len().min(fake_feat[i].len()) {
            let d = real_feat[i][j] - fake_feat[i][j];
            loss += d * d;
            count += 1;
        }
    }
    if count > 0 {
        loss / count as f32
    } else {
        0.0
    }
}

pub fn minibatch_std_dev(inp: &TMatrix) -> TMatrix {
    let batch = inp.len();
    if batch == 0 {
        return vec![];
    }
    let features = inp[0].len();

    // Compute mean per feature
    let mut mean = vec![0.0f32; features];
    for i in 0..batch {
        for j in 0..features {
            mean[j] += inp[i][j];
        }
    }
    for j in 0..features {
        mean[j] /= batch as f32;
    }

    // Compute std dev per feature
    let mut std_dev = vec![0.0f32; features];
    for i in 0..batch {
        for j in 0..features {
            let d = inp[i][j] - mean[j];
            std_dev[j] += d * d;
        }
    }
    for j in 0..features {
        std_dev[j] = (std_dev[j] / batch as f32).sqrt();
    }

    // Average std dev across features
    let avg_std: f32 = std_dev.iter().sum::<f32>() / features as f32;

    // Append as extra feature
    let mut result = crate::matrix::create_matrix(batch as i32, (features + 1) as i32);
    for i in 0..batch {
        for j in 0..features {
            result[i][j] = inp[i][j];
        }
        result[i][features] = avg_std;
    }
    result
}

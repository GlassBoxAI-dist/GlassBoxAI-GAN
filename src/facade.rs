/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * GANFacade — the unified facade API matching gan_facade_cuda.cu.
 * All GF_ prefixed functions are re-exported here.
 */

use crate::activations::*;
use crate::attention::*;
use crate::convolution::*;
use crate::layer::*;
use crate::loss::*;
use crate::matrix::*;
use crate::network::*;
use crate::normalization::*;
use crate::optimizer;
use crate::random::*;
use crate::security::*;
use crate::training::*;
use crate::types::*;

// =========================================================================
// GF_Op_ : LOW-LEVEL OPERATIONS
// =========================================================================

// --- Matrix ---
pub fn gf_op_create_matrix(rows: i32, cols: i32) -> TMatrix {
    create_matrix(rows, cols)
}
pub fn gf_op_create_vector(size: i32) -> TVector {
    create_vector(size)
}
pub fn gf_op_matrix_multiply(a: &TMatrix, b: &TMatrix) -> TMatrix {
    matrix_multiply(a, b)
}
pub fn gf_op_matrix_add(a: &TMatrix, b: &TMatrix) -> TMatrix {
    matrix_add(a, b)
}
pub fn gf_op_matrix_subtract(a: &TMatrix, b: &TMatrix) -> TMatrix {
    matrix_subtract(a, b)
}
pub fn gf_op_matrix_scale(a: &TMatrix, s: f32) -> TMatrix {
    matrix_scale(a, s)
}
pub fn gf_op_matrix_transpose(a: &TMatrix) -> TMatrix {
    matrix_transpose(a)
}
pub fn gf_op_matrix_normalize(a: &TMatrix) -> TMatrix {
    matrix_normalize(a)
}
pub fn gf_op_matrix_element_mul(a: &TMatrix, b: &TMatrix) -> TMatrix {
    matrix_element_mul(a, b)
}
pub fn gf_op_matrix_add_in_place(a: &mut TMatrix, b: &TMatrix) {
    matrix_add_in_place(a, b);
}
pub fn gf_op_matrix_scale_in_place(a: &mut TMatrix, s: f32) {
    matrix_scale_in_place(a, s);
}
pub fn gf_op_matrix_clip_in_place(a: &mut TMatrix, lo: f32, hi: f32) {
    matrix_clip_in_place(a, lo, hi);
}
pub fn gf_op_safe_get(m: &TMatrix, r: i32, c: i32, def: f32) -> f32 {
    safe_get(m, r, c, def)
}
pub fn gf_op_safe_set(m: &mut TMatrix, r: i32, c: i32, val: f32) {
    safe_set(m, r, c, val);
}

// --- Activations ---
pub fn gf_op_relu(a: &TMatrix) -> TMatrix {
    matrix_relu(a)
}
pub fn gf_op_leaky_relu(a: &TMatrix, alpha: f32) -> TMatrix {
    matrix_leaky_relu(a, alpha)
}
pub fn gf_op_sigmoid(a: &TMatrix) -> TMatrix {
    matrix_sigmoid(a)
}
pub fn gf_op_tanh(a: &TMatrix) -> TMatrix {
    matrix_tanh(a)
}
pub fn gf_op_softmax(a: &TMatrix) -> TMatrix {
    matrix_softmax(a)
}
pub fn gf_op_activate(a: &TMatrix, act: ActivationType) -> TMatrix {
    apply_activation(a, act)
}
pub fn gf_op_activation_backward(
    grad_out: &TMatrix,
    pre_act: &TMatrix,
    act: ActivationType,
) -> TMatrix {
    activation_backward(grad_out, pre_act, act)
}

// --- Convolution ---
pub fn gf_op_conv2d(inp: &TMatrix, layer: &mut Layer) -> TMatrix {
    conv2d_forward(inp, layer)
}
pub fn gf_op_conv2d_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    conv2d_backward(layer, grad_out)
}
pub fn gf_op_deconv2d(inp: &TMatrix, layer: &mut Layer) -> TMatrix {
    deconv2d_forward(inp, layer)
}
pub fn gf_op_deconv2d_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    deconv2d_backward(layer, grad_out)
}
pub fn gf_op_conv1d(inp: &TMatrix, layer: &mut Layer) -> TMatrix {
    conv1d_forward(inp, layer)
}
pub fn gf_op_conv1d_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    conv1d_backward(layer, grad_out)
}

// --- Normalization ---
pub fn gf_op_batch_norm(inp: &TMatrix, layer: &mut Layer) -> TMatrix {
    batch_norm_forward(inp, layer)
}
pub fn gf_op_batch_norm_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    batch_norm_backward(layer, grad_out)
}
pub fn gf_op_layer_norm(inp: &TMatrix, layer: &mut Layer) -> TMatrix {
    layer_norm_forward(inp, layer)
}
pub fn gf_op_layer_norm_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    layer_norm_backward(layer, grad_out)
}
pub fn gf_op_spectral_norm(layer: &mut Layer) -> TMatrix {
    spectral_normalize(layer)
}

// --- Attention ---
pub fn gf_op_attention(inp: &TMatrix, layer: &mut Layer) -> TMatrix {
    self_attention_forward(inp, layer)
}
pub fn gf_op_attention_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    self_attention_backward(layer, grad_out)
}

// --- Layer creation ---
pub fn gf_op_create_dense_layer(in_sz: i32, out_sz: i32, act: ActivationType) -> Layer {
    create_dense_layer(in_sz, out_sz, act)
}
pub fn gf_op_create_conv2d_layer(
    in_ch: i32,
    out_ch: i32,
    k_sz: i32,
    st: i32,
    pad: i32,
    w: i32,
    h: i32,
    act: ActivationType,
) -> Layer {
    create_conv2d_layer(in_ch, out_ch, k_sz, st, pad, w, h, act)
}
pub fn gf_op_create_deconv2d_layer(
    in_ch: i32,
    out_ch: i32,
    k_sz: i32,
    st: i32,
    pad: i32,
    w: i32,
    h: i32,
    act: ActivationType,
) -> Layer {
    create_deconv2d_layer(in_ch, out_ch, k_sz, st, pad, w, h, act)
}
pub fn gf_op_create_conv1d_layer(
    in_ch: i32,
    out_ch: i32,
    k_sz: i32,
    st: i32,
    pad: i32,
    in_len: i32,
    act: ActivationType,
) -> Layer {
    create_conv1d_layer(in_ch, out_ch, k_sz, st, pad, in_len, act)
}
pub fn gf_op_create_batch_norm_layer(features: i32) -> Layer {
    create_batch_norm_layer(features)
}
pub fn gf_op_create_layer_norm_layer(features: i32) -> Layer {
    create_layer_norm_layer(features)
}
pub fn gf_op_create_attention_layer(d_model: i32, n_heads: i32) -> Layer {
    create_attention_layer(d_model, n_heads)
}

// --- Layer dispatch ---
pub fn gf_op_layer_forward(layer: &mut Layer, inp: &TMatrix) -> TMatrix {
    layer_forward(layer, inp)
}
pub fn gf_op_layer_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    layer_backward(layer, grad_out)
}
pub fn gf_op_init_layer_optimizer(layer: &mut Layer, opt: Optimizer) {
    init_layer_optimizer(layer, opt);
}

// --- Random / Noise ---
pub fn gf_op_random_gaussian() -> f32 {
    random_gaussian()
}
pub fn gf_op_random_uniform(lo: f32, hi: f32) -> f32 {
    random_uniform(lo, hi)
}
pub fn gf_op_generate_noise(noise: &mut TMatrix, size: i32, depth: i32, nt: NoiseType) {
    generate_noise(noise, size, depth, nt);
}
pub fn gf_op_noise_slerp(v1: &TVector, v2: &TVector, t: f32) -> TVector {
    noise_slerp(v1, v2, t)
}

// =========================================================================
// GF_Gen_ : GENERATOR ACTIONS
// =========================================================================

pub fn gf_gen_build(sizes: &[i32], act: ActivationType, opt: Optimizer, lr: f32) -> Network {
    create_network(sizes, act, opt, lr)
}
pub fn gf_gen_build_conv(
    noise_dim: i32,
    cond_sz: i32,
    base_ch: i32,
    act: ActivationType,
    opt: Optimizer,
    lr: f32,
) -> Network {
    create_conv_generator(noise_dim, cond_sz, base_ch, act, opt, lr)
}
pub fn gf_gen_forward(gen: &mut Network, inp: &TMatrix) -> TMatrix {
    network_forward(gen, inp)
}
pub fn gf_gen_backward(gen: &mut Network, grad_out: &TMatrix) -> TMatrix {
    network_backward(gen, grad_out)
}
pub fn gf_gen_sample(gen: &mut Network, count: i32, noise_dim: i32, nt: NoiseType) -> TMatrix {
    crate::network::gf_gen_sample(gen, count, noise_dim, nt)
}
pub fn gf_gen_sample_conditional(
    gen: &mut Network,
    count: i32,
    noise_dim: i32,
    cond_sz: i32,
    nt: NoiseType,
    cond: &TMatrix,
) -> TMatrix {
    crate::network::gf_gen_sample_conditional(gen, count, noise_dim, cond_sz, nt, cond)
}
pub fn gf_gen_update_weights(gen: &mut Network) {
    network_update_weights(gen);
}
pub fn gf_gen_add_progressive_layer(gen: &mut Network, res_lvl: i32) {
    add_progressive_layer(gen, res_lvl, true);
}
pub fn gf_gen_get_layer_output(gen: &Network, idx: i32) -> TMatrix {
    get_layer_output(gen, idx)
}
pub fn gf_gen_set_training(gen: &mut Network, training: bool) {
    set_network_training(gen, training);
}
pub fn gf_gen_noise(size: i32, depth: i32, nt: NoiseType) -> TMatrix {
    crate::network::gf_gen_noise(size, depth, nt)
}
pub fn gf_gen_noise_slerp(v1: &TVector, v2: &TVector, t: f32) -> TVector {
    noise_slerp(v1, v2, t)
}
pub fn gf_gen_deep_copy(gen: &Network) -> Network {
    deep_copy_network(gen)
}

// =========================================================================
// GF_Disc_ : DISCRIMINATOR ACTIONS
// =========================================================================

pub fn gf_disc_build(sizes: &[i32], act: ActivationType, opt: Optimizer, lr: f32) -> Network {
    create_network(sizes, act, opt, lr)
}
pub fn gf_disc_build_conv(
    in_ch: i32,
    in_w: i32,
    in_h: i32,
    cond_sz: i32,
    base_ch: i32,
    act: ActivationType,
    opt: Optimizer,
    lr: f32,
) -> Network {
    create_conv_discriminator(in_ch, in_w, in_h, cond_sz, base_ch, act, opt, lr)
}
pub fn gf_disc_evaluate(disc: &mut Network, inp: &TMatrix) -> TMatrix {
    network_forward(disc, inp)
}
pub fn gf_disc_forward(disc: &mut Network, inp: &TMatrix) -> TMatrix {
    network_forward(disc, inp)
}
pub fn gf_disc_backward(disc: &mut Network, grad_out: &TMatrix) -> TMatrix {
    network_backward(disc, grad_out)
}
pub fn gf_disc_update_weights(disc: &mut Network) {
    network_update_weights(disc);
}
pub fn gf_disc_grad_penalty(
    disc: &mut Network,
    real: &TMatrix,
    fake: &TMatrix,
    lambda: f32,
) -> f32 {
    compute_gradient_penalty(disc, real, fake, lambda)
}
pub fn gf_disc_feature_match(
    disc: &mut Network,
    real: &TMatrix,
    fake: &TMatrix,
    feat_layer: i32,
) -> f32 {
    feature_matching_loss(disc, real, fake, feat_layer)
}
pub fn gf_disc_minibatch_std_dev(inp: &TMatrix) -> TMatrix {
    minibatch_std_dev(inp)
}
pub fn gf_disc_add_progressive_layer(disc: &mut Network, res_lvl: i32) {
    add_progressive_layer(disc, res_lvl, false);
}
pub fn gf_disc_get_layer_output(disc: &Network, idx: i32) -> TMatrix {
    get_layer_output(disc, idx)
}
pub fn gf_disc_set_training(disc: &mut Network, training: bool) {
    set_network_training(disc, training);
}
pub fn gf_disc_deep_copy(disc: &Network) -> Network {
    deep_copy_network(disc)
}

// =========================================================================
// GF_Train_ : TRAINING CONTROL
// =========================================================================

pub fn gf_train_full(gen: &mut Network, disc: &mut Network, ds: &Dataset, cfg: &GANConfig) -> GANMetrics {
    train_full(gen, disc, ds, cfg)
}
pub fn gf_train_step(
    gen: &mut Network,
    disc: &mut Network,
    real_batch: &TMatrix,
    noise: &TMatrix,
    cfg: &GANConfig,
) -> GANMetrics {
    train_step(gen, disc, real_batch, noise, cfg)
}
pub fn gf_train_optimize(net: &mut Network) {
    network_update_weights(net);
}
pub fn gf_train_adam_update(
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
    optimizer::adam_update_matrix(p, g, m, v, t, lr, b1, b2, eps, wd);
}
pub fn gf_train_sgd_update(p: &mut TMatrix, g: &TMatrix, lr: f32, wd: f32) {
    optimizer::sgd_update_matrix(p, g, lr, wd);
}
pub fn gf_train_rmsprop_update(
    p: &mut TMatrix,
    g: &TMatrix,
    cache: &mut TMatrix,
    lr: f32,
    decay: f32,
    eps: f32,
    wd: f32,
) {
    optimizer::rmsprop_update_matrix(p, g, cache, lr, decay, eps, wd);
}
pub fn gf_train_cosine_anneal(epoch: i32, max_ep: i32, base_lr: f32, min_lr: f32) -> f32 {
    optimizer::cosine_anneal(epoch, max_ep, base_lr, min_lr)
}

// --- Loss ---
pub fn gf_train_bce_loss(pred: &TMatrix, target: &TMatrix) -> f32 {
    binary_cross_entropy(pred, target)
}
pub fn gf_train_bce_grad(pred: &TMatrix, target: &TMatrix) -> TMatrix {
    bce_gradient(pred, target)
}
pub fn gf_train_wgan_disc_loss(d_real: &TMatrix, d_fake: &TMatrix) -> f32 {
    wgan_disc_loss(d_real, d_fake)
}
pub fn gf_train_wgan_gen_loss(d_fake: &TMatrix) -> f32 {
    wgan_gen_loss(d_fake)
}
pub fn gf_train_hinge_disc_loss(d_real: &TMatrix, d_fake: &TMatrix) -> f32 {
    hinge_disc_loss(d_real, d_fake)
}
pub fn gf_train_hinge_gen_loss(d_fake: &TMatrix) -> f32 {
    hinge_gen_loss(d_fake)
}
pub fn gf_train_ls_disc_loss(d_real: &TMatrix, d_fake: &TMatrix) -> f32 {
    ls_disc_loss(d_real, d_fake)
}
pub fn gf_train_ls_gen_loss(d_fake: &TMatrix) -> f32 {
    ls_gen_loss(d_fake)
}
pub fn gf_train_label_smoothing(labels: &TMatrix, lo: f32, hi: f32) -> TMatrix {
    apply_label_smoothing(labels, lo, hi)
}

// --- Data ---
pub fn gf_train_load_dataset(path: &str, dt: DataType) -> Dataset {
    load_dataset(path, dt)
}
pub fn gf_train_load_bmp(path: &str) -> Dataset {
    load_bmp_dataset(path)
}
pub fn gf_train_load_wav(path: &str) -> Dataset {
    load_wav_dataset(path)
}
pub fn gf_train_create_synthetic(count: i32, features: i32) -> Dataset {
    create_synthetic_dataset(count, features)
}
pub fn gf_train_augment(sample: &TMatrix, dt: DataType) -> TMatrix {
    augment_sample(sample, dt)
}

// --- Metrics ---
pub fn gf_train_compute_fid(real_s: &TMatrixArray, fake_s: &TMatrixArray) -> f32 {
    compute_fid(real_s, fake_s)
}
pub fn gf_train_compute_is(samples: &TMatrixArray) -> f32 {
    compute_is(samples)
}
pub fn gf_train_log_metrics(met: &GANMetrics, filename: &str) {
    log_metrics(met, filename);
}

// --- I/O ---
pub fn gf_train_save_model(net: &Network, filename: &str) {
    save_network_binary(net, filename);
}
pub fn gf_train_load_model(net: &mut Network, filename: &str) {
    load_network_binary(net, filename);
}
pub fn gf_train_save_json(gen: &Network, disc: &Network, filename: &str) {
    save_gan_to_json(gen, disc, filename);
}
pub fn gf_train_load_json(gen: &mut Network, disc: &mut Network, filename: &str) {
    load_gan_from_json(gen, disc, filename);
}
pub fn gf_train_save_checkpoint(gen: &Network, disc: &Network, ep: i32, dir: &str) {
    save_checkpoint(gen, disc, ep, dir);
}
pub fn gf_train_load_checkpoint(gen: &mut Network, disc: &mut Network, ep: i32, dir: &str) {
    load_checkpoint(gen, disc, ep, dir);
}
pub fn gf_train_save_samples(
    gen: &mut Network,
    ep: i32,
    dir: &str,
    noise_dim: i32,
    nt: NoiseType,
) {
    save_generated_samples(gen, ep, dir, noise_dim, nt);
}
pub fn gf_train_plot_csv(filename: &str, d_loss: &[f32], g_loss: &[f32], cnt: i32) {
    plot_loss_csv(filename, d_loss, g_loss, cnt);
}
pub fn gf_train_print_bar(d_loss: f32, g_loss: f32, width: i32) {
    print_loss_bar(d_loss, g_loss, width);
}

// =========================================================================
// GF_Sec_ : SECURITY & ENTROPY
// =========================================================================

pub fn gf_sec_audit_log(msg: &str, log_file: &str) {
    audit_log(msg, log_file);
}
pub fn gf_sec_secure_randomize() {
    secure_randomize();
}
pub fn gf_sec_get_os_random() -> u8 {
    secure_random_byte()
}
pub fn gf_sec_validate_path(path: &str) -> bool {
    validate_path(path)
}
pub fn gf_sec_verify_weights(layer: &mut Layer) {
    validate_and_clean_weights(layer);
}
pub fn gf_sec_verify_network(net: &mut Network) {
    verify_network(net);
}
pub fn gf_sec_encrypt_model(in_f: &str, out_f: &str, key: &str) {
    encrypt_file(in_f, out_f, key);
}
pub fn gf_sec_decrypt_model(in_f: &str, out_f: &str, key: &str) {
    decrypt_file(in_f, out_f, key);
}
pub fn gf_sec_bounds_check(m: &TMatrix, r: i32, c: i32) -> bool {
    bounds_check(m, r, c)
}
pub fn gf_sec_run_tests() -> bool {
    crate::security::run_tests()
}
pub fn gf_sec_run_fuzz_tests(iterations: i32) -> bool {
    crate::security::run_fuzz_tests(iterations)
}

// =========================================================================
// GF_Run : HIGH-LEVEL ORCHESTRATION
// =========================================================================

/// Build, train, and return a fully-trained GAN from a single config.
/// This is the library-callable equivalent of the CLI's main() training block.
pub fn gf_run(config: &GANConfig) -> GANResult {
    // Build generator and discriminator
    let (mut generator, mut discriminator) = if config.use_conv {
        let gen = gf_gen_build_conv(
            config.noise_depth, config.condition_size, 8,
            config.activation, config.optimizer, config.learning_rate,
        );
        let disc = gf_disc_build_conv(
            1, 32, 32, config.condition_size, 8,
            config.activation, config.optimizer, config.learning_rate,
        );
        (gen, disc)
    } else {
        let gen_sizes = vec![config.noise_depth + config.condition_size, 128, 64, 1];
        let disc_sizes = vec![1, 64, 128, 1];
        let gen  = gf_gen_build(&gen_sizes,  config.activation, config.optimizer, config.learning_rate);
        let disc = gf_disc_build(&disc_sizes, config.activation, config.optimizer, config.learning_rate);
        (gen, disc)
    };

    // Apply spectral norm u/v vectors to discriminator dense layers
    if config.use_spectral_norm {
        for i in 0..discriminator.layer_count as usize {
            if discriminator.layers[i].layer_type == LayerType::Dense {
                let rows = discriminator.layers[i].weights.len();
                let cols = if rows > 0 { discriminator.layers[i].weights[0].len() } else { 0 };
                discriminator.layers[i].spectral_u = (0..rows).map(|_| random_gaussian()).collect();
                discriminator.layers[i].spectral_v = (0..cols).map(|_| random_gaussian()).collect();
            }
        }
    }

    // TTUR per-network learning rates
    if config.generator_lr > 0.0     { generator.learning_rate     = config.generator_lr;     }
    if config.discriminator_lr > 0.0 { discriminator.learning_rate = config.discriminator_lr; }
    if config.use_weight_decay {
        generator.weight_decay     = config.weight_decay_val;
        discriminator.weight_decay = config.weight_decay_val;
    }

    // Verify weight integrity
    gf_sec_verify_network(&mut generator);
    gf_sec_verify_network(&mut discriminator);

    // Load pretrained weights if requested
    if !config.load_model.is_empty() {
        println!("Loading model: {}", config.load_model);
        gf_train_load_model(&mut generator, &config.load_model);
    }
    if !config.load_json_model.is_empty() {
        println!("Loading JSON model: {}", config.load_json_model);
        gf_train_load_json(&mut generator, &mut discriminator, &config.load_json_model);
    }

    // Load or generate dataset
    let dataset = if !config.data_path.is_empty() {
        println!("Loading dataset from: {}", config.data_path);
        gf_train_load_dataset(&config.data_path, config.data_type)
    } else if config.use_conv {
        println!("Generating synthetic image dataset...");
        gf_train_create_synthetic(50, 32 * 32)
    } else {
        println!("Generating synthetic dataset...");
        gf_train_create_synthetic(1000, 1)
    };
    println!("Dataset: {} samples\n", dataset.count);

    // Create output directory
    if !config.output_dir.is_empty() {
        std::fs::create_dir_all(&config.output_dir).ok();
    }

    // Train
    println!("Starting training...\n");
    let metrics = gf_train_full(&mut generator, &mut discriminator, &dataset, config);

    // Save model
    if !config.save_model.is_empty() {
        if config.save_model.ends_with(".json") || config.save_model.ends_with(".JSON") {
            gf_train_save_json(&generator, &discriminator, &config.save_model);
        } else {
            gf_train_save_model(&generator, &config.save_model);
            let disc_fn = if let Some(pos) = config.save_model.find(".bin") {
                format!("{}_disc.bin", &config.save_model[..pos])
            } else {
                format!("{}_disc", config.save_model)
            };
            gf_train_save_model(&discriminator, &disc_fn);
        }
        if config.use_encryption && !config.encryption_key.is_empty() {
            let enc_path = format!("{}.enc", config.save_model);
            gf_sec_encrypt_model(&config.save_model, &enc_path, &config.encryption_key);
            println!("Model encrypted to: {}", enc_path);
        }
    }

    if config.audit_log {
        gf_sec_audit_log("GAN completed successfully", &config.audit_log_file);
    }

    GANResult { generator, discriminator, metrics }
}

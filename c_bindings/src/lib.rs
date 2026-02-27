/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * C FFI bindings for facaded_gan_cuda.
 * All types are opaque; callers use accessor functions and must call
 * the matching gf_*_free() for every allocated object.
 *
 * Matrix layout: row-major flat f32 array (row * cols + col).
 * Strings:       NUL-terminated UTF-8 const char*.
 * Booleans:      int  (0 = false, non-zero = true).
 */

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_int};

use facaded_gan_cuda::types::*;
use facaded_gan_cuda::facade;
use facaded_gan_cuda::backend::{self, ComputeBackend};

// =========================================================================
// Internal helpers
// =========================================================================

/// Convert a *const c_char to &str, returning "" on null or invalid UTF-8.
unsafe fn cstr(s: *const c_char) -> &'static str {
    if s.is_null() { return ""; }
    CStr::from_ptr(s).to_str().unwrap_or("")
}

// =========================================================================
// Opaque structs
// =========================================================================

pub struct GanMatrix  { flat: Vec<f32>, rows: i32, cols: i32 }
pub struct GanVector  { data: Vec<f32> }
pub struct GanNetwork { inner: Network }
pub struct GanDataset { inner: Dataset }
pub struct GanConfig  { inner: GANConfig }
pub struct GanMetrics { inner: GANMetrics }
pub struct GanResult  { inner: GANResult }

// =========================================================================
// Internal converters
// =========================================================================

fn matrix_to_gan(m: &TMatrix) -> *mut GanMatrix {
    let rows = m.len() as i32;
    let cols = if rows > 0 { m[0].len() as i32 } else { 0 };
    let mut flat = Vec::with_capacity((rows * cols) as usize);
    for row in m {
        flat.extend_from_slice(row);
        for _ in row.len()..cols as usize { flat.push(0.0); }
    }
    Box::into_raw(Box::new(GanMatrix { flat, rows, cols }))
}

fn vector_to_gan(v: &TVector) -> *mut GanVector {
    Box::into_raw(Box::new(GanVector { data: v.clone() }))
}

unsafe fn gan_to_matrix(m: *const GanMatrix) -> TMatrix {
    if m.is_null() { return vec![]; }
    let m = &*m;
    let (r, c) = (m.rows as usize, m.cols as usize);
    (0..r).map(|i| m.flat[i*c..(i+1)*c].to_vec()).collect()
}

unsafe fn gan_to_vector(v: *const GanVector) -> TVector {
    if v.is_null() { return vec![]; }
    (*v).data.clone()
}

fn parse_activation(s: &str) -> ActivationType {
    match s { "relu" => ActivationType::ReLU, "sigmoid" => ActivationType::Sigmoid,
              "tanh" => ActivationType::Tanh, "leaky"   => ActivationType::LeakyReLU,
              _      => ActivationType::None }
}
fn parse_optimizer(s: &str) -> Optimizer {
    match s { "sgd" => Optimizer::SGD, "rmsprop" => Optimizer::RMSProp, _ => Optimizer::Adam }
}
fn parse_noise_type(s: &str) -> NoiseType {
    match s { "uniform" => NoiseType::Uniform, "analog" => NoiseType::Analog, _ => NoiseType::Gauss }
}
fn parse_data_type(s: &str) -> DataType {
    match s { "image" => DataType::Image, "audio" => DataType::Audio, _ => DataType::Vector }
}
fn parse_loss_type(s: &str) -> LossType {
    match s { "wgan" | "wgan-gp" => LossType::WGANGP, "hinge" => LossType::Hinge,
              "ls" => LossType::LeastSquares, _ => LossType::BCE }
}
fn parse_backend(s: &str) -> ComputeBackend {
    s.parse().unwrap_or(ComputeBackend::CPU)
}

// =========================================================================
// Matrix API
// =========================================================================

/// Create a zero-initialised rows×cols matrix.  Free with gf_matrix_free().
#[no_mangle]
pub extern "C" fn gf_matrix_create(rows: c_int, cols: c_int) -> *mut GanMatrix {
    if rows <= 0 || cols <= 0 { return std::ptr::null_mut(); }
    Box::into_raw(Box::new(GanMatrix {
        flat: vec![0.0f32; (rows * cols) as usize],
        rows, cols,
    }))
}

/// Create a matrix from a caller-supplied flat row-major array (copied).
#[no_mangle]
pub unsafe extern "C" fn gf_matrix_from_data(
    data: *const c_float, rows: c_int, cols: c_int,
) -> *mut GanMatrix {
    if data.is_null() || rows <= 0 || cols <= 0 { return std::ptr::null_mut(); }
    let len = (rows * cols) as usize;
    let flat = std::slice::from_raw_parts(data, len).to_vec();
    Box::into_raw(Box::new(GanMatrix { flat, rows, cols }))
}

/// Free a GanMatrix returned by any gf_* function.
#[no_mangle]
pub unsafe extern "C" fn gf_matrix_free(m: *mut GanMatrix) {
    if !m.is_null() { drop(Box::from_raw(m)); }
}

/// Pointer to the flat row-major data (valid until gf_matrix_free).
#[no_mangle]
pub unsafe extern "C" fn gf_matrix_data(m: *const GanMatrix) -> *const c_float {
    if m.is_null() { return std::ptr::null(); }
    (*m).flat.as_ptr()
}
#[no_mangle]
pub unsafe extern "C" fn gf_matrix_rows(m: *const GanMatrix) -> c_int {
    if m.is_null() { return 0; } (*m).rows
}
#[no_mangle]
pub unsafe extern "C" fn gf_matrix_cols(m: *const GanMatrix) -> c_int {
    if m.is_null() { return 0; } (*m).cols
}
#[no_mangle]
pub unsafe extern "C" fn gf_matrix_get(m: *const GanMatrix, row: c_int, col: c_int) -> c_float {
    if m.is_null() { return 0.0; }
    let m = &*m;
    if row < 0 || row >= m.rows || col < 0 || col >= m.cols { return 0.0; }
    m.flat[(row * m.cols + col) as usize]
}
#[no_mangle]
pub unsafe extern "C" fn gf_matrix_set(m: *mut GanMatrix, row: c_int, col: c_int, val: c_float) {
    if m.is_null() { return; }
    let m = &mut *m;
    if row < 0 || row >= m.rows || col < 0 || col >= m.cols { return; }
    m.flat[(row * m.cols + col) as usize] = val;
}

#[no_mangle]
pub unsafe extern "C" fn gf_matrix_multiply(a: *const GanMatrix, b: *const GanMatrix) -> *mut GanMatrix {
    matrix_to_gan(&facaded_gan_cuda::matrix::matrix_multiply(&gan_to_matrix(a), &gan_to_matrix(b)))
}
#[no_mangle]
pub unsafe extern "C" fn gf_matrix_add(a: *const GanMatrix, b: *const GanMatrix) -> *mut GanMatrix {
    matrix_to_gan(&facaded_gan_cuda::matrix::matrix_add(&gan_to_matrix(a), &gan_to_matrix(b)))
}
#[no_mangle]
pub unsafe extern "C" fn gf_matrix_subtract(a: *const GanMatrix, b: *const GanMatrix) -> *mut GanMatrix {
    matrix_to_gan(&facaded_gan_cuda::matrix::matrix_subtract(&gan_to_matrix(a), &gan_to_matrix(b)))
}
#[no_mangle]
pub unsafe extern "C" fn gf_matrix_scale(a: *const GanMatrix, s: c_float) -> *mut GanMatrix {
    matrix_to_gan(&facaded_gan_cuda::matrix::matrix_scale(&gan_to_matrix(a), s))
}
#[no_mangle]
pub unsafe extern "C" fn gf_matrix_transpose(a: *const GanMatrix) -> *mut GanMatrix {
    matrix_to_gan(&facaded_gan_cuda::matrix::matrix_transpose(&gan_to_matrix(a)))
}
#[no_mangle]
pub unsafe extern "C" fn gf_matrix_normalize(a: *const GanMatrix) -> *mut GanMatrix {
    matrix_to_gan(&facaded_gan_cuda::matrix::matrix_normalize(&gan_to_matrix(a)))
}
#[no_mangle]
pub unsafe extern "C" fn gf_matrix_element_mul(a: *const GanMatrix, b: *const GanMatrix) -> *mut GanMatrix {
    matrix_to_gan(&facaded_gan_cuda::matrix::matrix_element_mul(&gan_to_matrix(a), &gan_to_matrix(b)))
}
#[no_mangle]
pub unsafe extern "C" fn gf_matrix_safe_get(m: *const GanMatrix, r: c_int, c: c_int, def: c_float) -> c_float {
    facaded_gan_cuda::matrix::safe_get(&gan_to_matrix(m), r, c, def)
}

// =========================================================================
// Vector API
// =========================================================================

#[no_mangle]
pub extern "C" fn gf_vector_create(len: c_int) -> *mut GanVector {
    if len <= 0 { return std::ptr::null_mut(); }
    Box::into_raw(Box::new(GanVector { data: vec![0.0f32; len as usize] }))
}
#[no_mangle]
pub unsafe extern "C" fn gf_vector_free(v: *mut GanVector) {
    if !v.is_null() { drop(Box::from_raw(v)); }
}
#[no_mangle]
pub unsafe extern "C" fn gf_vector_data(v: *const GanVector) -> *const c_float {
    if v.is_null() { return std::ptr::null(); } (*v).data.as_ptr()
}
#[no_mangle]
pub unsafe extern "C" fn gf_vector_len(v: *const GanVector) -> c_int {
    if v.is_null() { return 0; } (*v).data.len() as c_int
}
#[no_mangle]
pub unsafe extern "C" fn gf_vector_get(v: *const GanVector, idx: c_int) -> c_float {
    if v.is_null() { return 0.0; }
    let v = &*v;
    if idx < 0 || idx as usize >= v.data.len() { return 0.0; }
    v.data[idx as usize]
}

/// Spherical interpolation between two noise vectors.  Free result with gf_vector_free().
#[no_mangle]
pub unsafe extern "C" fn gf_vector_noise_slerp(
    v1: *const GanVector, v2: *const GanVector, t: c_float,
) -> *mut GanVector {
    vector_to_gan(&facade::gf_op_noise_slerp(&gan_to_vector(v1), &gan_to_vector(v2), t))
}

// =========================================================================
// Config API
// =========================================================================

/// Create a GANConfig with default values.  Free with gf_config_free().
#[no_mangle]
pub extern "C" fn gf_config_create() -> *mut GanConfig {
    Box::into_raw(Box::new(GanConfig { inner: GANConfig::default() }))
}
#[no_mangle]
pub unsafe extern "C" fn gf_config_free(cfg: *mut GanConfig) {
    if !cfg.is_null() { drop(Box::from_raw(cfg)); }
}

// -- integer fields --
macro_rules! cfg_int {
    ($get:ident, $set:ident, $field:ident) => {
        #[no_mangle] pub unsafe extern "C" fn $get(c: *const GanConfig) -> c_int {
            if c.is_null() { return 0; } (*c).inner.$field
        }
        #[no_mangle] pub unsafe extern "C" fn $set(c: *mut GanConfig, v: c_int) {
            if !c.is_null() { (*c).inner.$field = v; }
        }
    }
}
cfg_int!(gf_config_get_epochs,              gf_config_set_epochs,              epochs);
cfg_int!(gf_config_get_batch_size,          gf_config_set_batch_size,          batch_size);
cfg_int!(gf_config_get_noise_depth,         gf_config_set_noise_depth,         noise_depth);
cfg_int!(gf_config_get_condition_size,      gf_config_set_condition_size,      condition_size);
cfg_int!(gf_config_get_generator_bits,      gf_config_set_generator_bits,      generator_bits);
cfg_int!(gf_config_get_discriminator_bits,  gf_config_set_discriminator_bits,  discriminator_bits);
cfg_int!(gf_config_get_max_res_level,       gf_config_set_max_res_level,       max_res_level);
cfg_int!(gf_config_get_metric_interval,     gf_config_set_metric_interval,     metric_interval);
cfg_int!(gf_config_get_checkpoint_interval, gf_config_set_checkpoint_interval, checkpoint_interval);
cfg_int!(gf_config_get_fuzz_iterations,     gf_config_set_fuzz_iterations,     fuzz_iterations);
cfg_int!(gf_config_get_num_threads,         gf_config_set_num_threads,         num_threads);

// -- float fields --
macro_rules! cfg_float {
    ($get:ident, $set:ident, $field:ident) => {
        #[no_mangle] pub unsafe extern "C" fn $get(c: *const GanConfig) -> c_float {
            if c.is_null() { return 0.0; } (*c).inner.$field
        }
        #[no_mangle] pub unsafe extern "C" fn $set(c: *mut GanConfig, v: c_float) {
            if !c.is_null() { (*c).inner.$field = v; }
        }
    }
}
cfg_float!(gf_config_get_learning_rate,    gf_config_set_learning_rate,    learning_rate);
cfg_float!(gf_config_get_gp_lambda,        gf_config_set_gp_lambda,        gp_lambda);
cfg_float!(gf_config_get_generator_lr,     gf_config_set_generator_lr,     generator_lr);
cfg_float!(gf_config_get_discriminator_lr, gf_config_set_discriminator_lr, discriminator_lr);
cfg_float!(gf_config_get_weight_decay_val, gf_config_set_weight_decay_val, weight_decay_val);

// -- bool fields (exposed as int) --
macro_rules! cfg_bool {
    ($get:ident, $set:ident, $field:ident) => {
        #[no_mangle] pub unsafe extern "C" fn $get(c: *const GanConfig) -> c_int {
            if c.is_null() { return 0; } (*c).inner.$field as c_int
        }
        #[no_mangle] pub unsafe extern "C" fn $set(c: *mut GanConfig, v: c_int) {
            if !c.is_null() { (*c).inner.$field = v != 0; }
        }
    }
}
cfg_bool!(gf_config_get_use_batch_norm,       gf_config_set_use_batch_norm,       use_batch_norm);
cfg_bool!(gf_config_get_use_layer_norm,        gf_config_set_use_layer_norm,        use_layer_norm);
cfg_bool!(gf_config_get_use_spectral_norm,     gf_config_set_use_spectral_norm,     use_spectral_norm);
cfg_bool!(gf_config_get_use_label_smoothing,   gf_config_set_use_label_smoothing,   use_label_smoothing);
cfg_bool!(gf_config_get_use_feature_matching,  gf_config_set_use_feature_matching,  use_feature_matching);
cfg_bool!(gf_config_get_use_minibatch_std_dev, gf_config_set_use_minibatch_std_dev, use_minibatch_std_dev);
cfg_bool!(gf_config_get_use_progressive,       gf_config_set_use_progressive,       use_progressive);
cfg_bool!(gf_config_get_use_augmentation,      gf_config_set_use_augmentation,      use_augmentation);
cfg_bool!(gf_config_get_compute_metrics,       gf_config_set_compute_metrics,       compute_metrics);
cfg_bool!(gf_config_get_use_weight_decay,      gf_config_set_use_weight_decay,      use_weight_decay);
cfg_bool!(gf_config_get_use_cosine_anneal,     gf_config_set_use_cosine_anneal,     use_cosine_anneal);
cfg_bool!(gf_config_get_audit_log,             gf_config_set_audit_log,             audit_log);
cfg_bool!(gf_config_get_use_encryption,        gf_config_set_use_encryption,        use_encryption);
cfg_bool!(gf_config_get_use_conv,              gf_config_set_use_conv,              use_conv);
cfg_bool!(gf_config_get_use_attention,         gf_config_set_use_attention,         use_attention);

// -- string fields (caller supplies NUL-terminated; returned strings are static/internal) --
macro_rules! cfg_str_set {
    ($set:ident, $field:ident) => {
        #[no_mangle] pub unsafe extern "C" fn $set(c: *mut GanConfig, v: *const c_char) {
            if !c.is_null() { (*c).inner.$field = cstr(v).to_owned(); }
        }
    }
}
cfg_str_set!(gf_config_set_save_model,      save_model);
cfg_str_set!(gf_config_set_load_model,      load_model);
cfg_str_set!(gf_config_set_load_json_model, load_json_model);
cfg_str_set!(gf_config_set_output_dir,      output_dir);
cfg_str_set!(gf_config_set_data_path,       data_path);
cfg_str_set!(gf_config_set_audit_log_file,  audit_log_file);
cfg_str_set!(gf_config_set_encryption_key,  encryption_key);
cfg_str_set!(gf_config_set_patch_config,    patch_config);

// -- enum fields (string-based) --
#[no_mangle] pub unsafe extern "C" fn gf_config_set_activation(c: *mut GanConfig, v: *const c_char) {
    if !c.is_null() { (*c).inner.activation = parse_activation(cstr(v)); }
}
#[no_mangle] pub unsafe extern "C" fn gf_config_set_noise_type(c: *mut GanConfig, v: *const c_char) {
    if !c.is_null() { (*c).inner.noise_type = parse_noise_type(cstr(v)); }
}
#[no_mangle] pub unsafe extern "C" fn gf_config_set_optimizer(c: *mut GanConfig, v: *const c_char) {
    if !c.is_null() { (*c).inner.optimizer = parse_optimizer(cstr(v)); }
}
#[no_mangle] pub unsafe extern "C" fn gf_config_set_loss_type(c: *mut GanConfig, v: *const c_char) {
    if !c.is_null() { (*c).inner.loss_type = parse_loss_type(cstr(v)); }
}
#[no_mangle] pub unsafe extern "C" fn gf_config_set_data_type(c: *mut GanConfig, v: *const c_char) {
    if !c.is_null() { (*c).inner.data_type = parse_data_type(cstr(v)); }
}

// =========================================================================
// Network API
// =========================================================================

/// Build a dense generator.  sizes[] has num_sizes entries.  Free with gf_network_free().
#[no_mangle]
pub unsafe extern "C" fn gf_gen_build(
    sizes: *const c_int, num_sizes: c_int,
    act: *const c_char, opt: *const c_char, lr: c_float,
) -> *mut GanNetwork {
    if sizes.is_null() || num_sizes <= 0 { return std::ptr::null_mut(); }
    let sz: Vec<i32> = std::slice::from_raw_parts(sizes, num_sizes as usize).to_vec();
    let net = facade::gf_gen_build(&sz, parse_activation(cstr(act)), parse_optimizer(cstr(opt)), lr);
    Box::into_raw(Box::new(GanNetwork { inner: net }))
}

/// Build a convolutional generator.  Free with gf_network_free().
#[no_mangle]
pub unsafe extern "C" fn gf_gen_build_conv(
    noise_dim: c_int, cond_sz: c_int, base_ch: c_int,
    act: *const c_char, opt: *const c_char, lr: c_float,
) -> *mut GanNetwork {
    let net = facade::gf_gen_build_conv(noise_dim, cond_sz, base_ch,
        parse_activation(cstr(act)), parse_optimizer(cstr(opt)), lr);
    Box::into_raw(Box::new(GanNetwork { inner: net }))
}

/// Build a dense discriminator.  Free with gf_network_free().
#[no_mangle]
pub unsafe extern "C" fn gf_disc_build(
    sizes: *const c_int, num_sizes: c_int,
    act: *const c_char, opt: *const c_char, lr: c_float,
) -> *mut GanNetwork {
    if sizes.is_null() || num_sizes <= 0 { return std::ptr::null_mut(); }
    let sz: Vec<i32> = std::slice::from_raw_parts(sizes, num_sizes as usize).to_vec();
    let net = facade::gf_disc_build(&sz, parse_activation(cstr(act)), parse_optimizer(cstr(opt)), lr);
    Box::into_raw(Box::new(GanNetwork { inner: net }))
}

/// Build a convolutional discriminator.  Free with gf_network_free().
#[no_mangle]
pub unsafe extern "C" fn gf_disc_build_conv(
    in_ch: c_int, in_w: c_int, in_h: c_int, cond_sz: c_int, base_ch: c_int,
    act: *const c_char, opt: *const c_char, lr: c_float,
) -> *mut GanNetwork {
    let net = facade::gf_disc_build_conv(in_ch, in_w, in_h, cond_sz, base_ch,
        parse_activation(cstr(act)), parse_optimizer(cstr(opt)), lr);
    Box::into_raw(Box::new(GanNetwork { inner: net }))
}

#[no_mangle]
pub unsafe extern "C" fn gf_network_free(net: *mut GanNetwork) {
    if !net.is_null() { drop(Box::from_raw(net)); }
}

#[no_mangle]
pub unsafe extern "C" fn gf_network_layer_count(net: *const GanNetwork) -> c_int {
    if net.is_null() { return 0; } (*net).inner.layer_count
}
#[no_mangle]
pub unsafe extern "C" fn gf_network_learning_rate(net: *const GanNetwork) -> c_float {
    if net.is_null() { return 0.0; } (*net).inner.learning_rate
}
#[no_mangle]
pub unsafe extern "C" fn gf_network_is_training(net: *const GanNetwork) -> c_int {
    if net.is_null() { return 0; } (*net).inner.is_training as c_int
}

/// Forward pass.  inp is a batch_size×features matrix.  Free result with gf_matrix_free().
#[no_mangle]
pub unsafe extern "C" fn gf_network_forward(
    net: *mut GanNetwork, inp: *const GanMatrix,
) -> *mut GanMatrix {
    if net.is_null() { return std::ptr::null_mut(); }
    let out = facade::gf_gen_forward(&mut (*net).inner, &gan_to_matrix(inp));
    matrix_to_gan(&out)
}

/// Backward pass.  grad_out must have same shape as last forward output.
/// Free result with gf_matrix_free().
#[no_mangle]
pub unsafe extern "C" fn gf_network_backward(
    net: *mut GanNetwork, grad_out: *const GanMatrix,
) -> *mut GanMatrix {
    if net.is_null() { return std::ptr::null_mut(); }
    let g = facade::gf_gen_backward(&mut (*net).inner, &gan_to_matrix(grad_out));
    matrix_to_gan(&g)
}

#[no_mangle]
pub unsafe extern "C" fn gf_network_update_weights(net: *mut GanNetwork) {
    if !net.is_null() { facade::gf_gen_update_weights(&mut (*net).inner); }
}

#[no_mangle]
pub unsafe extern "C" fn gf_network_set_training(net: *mut GanNetwork, training: c_int) {
    if !net.is_null() { facade::gf_gen_set_training(&mut (*net).inner, training != 0); }
}

/// Sample count synthetic outputs.  Free result with gf_matrix_free().
#[no_mangle]
pub unsafe extern "C" fn gf_network_sample(
    net: *mut GanNetwork, count: c_int, noise_dim: c_int, noise_type: *const c_char,
) -> *mut GanMatrix {
    if net.is_null() { return std::ptr::null_mut(); }
    let out = facade::gf_gen_sample(&mut (*net).inner, count, noise_dim, parse_noise_type(cstr(noise_type)));
    matrix_to_gan(&out)
}

/// Verify and sanitise weights (clears NaN/Inf).
#[no_mangle]
pub unsafe extern "C" fn gf_network_verify(net: *mut GanNetwork) {
    if !net.is_null() { facade::gf_sec_verify_network(&mut (*net).inner); }
}

/// Save network weights to file (binary JSON).
#[no_mangle]
pub unsafe extern "C" fn gf_network_save(net: *const GanNetwork, path: *const c_char) {
    if net.is_null() { return; }
    facade::gf_train_save_model(&(*net).inner, cstr(path));
}

/// Load network weights from file (binary JSON).
#[no_mangle]
pub unsafe extern "C" fn gf_network_load(net: *mut GanNetwork, path: *const c_char) {
    if net.is_null() { return; }
    facade::gf_train_load_model(&mut (*net).inner, cstr(path));
}

// =========================================================================
// Dataset API
// =========================================================================

/// Create a synthetic dataset of count samples, each with features dimensions.
/// Free with gf_dataset_free().
#[no_mangle]
pub unsafe extern "C" fn gf_dataset_create_synthetic(count: c_int, features: c_int) -> *mut GanDataset {
    Box::into_raw(Box::new(GanDataset { inner: facade::gf_train_create_synthetic(count, features) }))
}

/// Load a dataset from path.  data_type: "image", "audio", or "vector".
/// Free with gf_dataset_free().
#[no_mangle]
pub unsafe extern "C" fn gf_dataset_load(path: *const c_char, data_type: *const c_char) -> *mut GanDataset {
    Box::into_raw(Box::new(GanDataset {
        inner: facade::gf_train_load_dataset(cstr(path), parse_data_type(cstr(data_type)))
    }))
}

#[no_mangle]
pub unsafe extern "C" fn gf_dataset_free(ds: *mut GanDataset) {
    if !ds.is_null() { drop(Box::from_raw(ds)); }
}

#[no_mangle]
pub unsafe extern "C" fn gf_dataset_count(ds: *const GanDataset) -> c_int {
    if ds.is_null() { return 0; } (*ds).inner.count
}

// =========================================================================
// Training API
// =========================================================================

/// Run one full training session (all epochs).  Free result with gf_metrics_free().
#[no_mangle]
pub unsafe extern "C" fn gf_train_full(
    gen: *mut GanNetwork, disc: *mut GanNetwork,
    ds: *const GanDataset, cfg: *const GanConfig,
) -> *mut GanMetrics {
    if gen.is_null() || disc.is_null() || ds.is_null() || cfg.is_null() {
        return std::ptr::null_mut();
    }
    let m = facade::gf_train_full(&mut (*gen).inner, &mut (*disc).inner,
                                   &(*ds).inner, &(*cfg).inner);
    Box::into_raw(Box::new(GanMetrics { inner: m }))
}

/// Run a single training step.  real_batch and noise are both batch×features matrices.
/// Free result with gf_metrics_free().
#[no_mangle]
pub unsafe extern "C" fn gf_train_step(
    gen: *mut GanNetwork, disc: *mut GanNetwork,
    real_batch: *const GanMatrix, noise: *const GanMatrix,
    cfg: *const GanConfig,
) -> *mut GanMetrics {
    if gen.is_null() || disc.is_null() || cfg.is_null() { return std::ptr::null_mut(); }
    let m = facade::gf_train_step(&mut (*gen).inner, &mut (*disc).inner,
                                   &gan_to_matrix(real_batch), &gan_to_matrix(noise),
                                   &(*cfg).inner);
    Box::into_raw(Box::new(GanMetrics { inner: m }))
}

/// Save both networks to a JSON file.
#[no_mangle]
pub unsafe extern "C" fn gf_train_save_json(
    gen: *const GanNetwork, disc: *const GanNetwork, path: *const c_char,
) {
    if gen.is_null() || disc.is_null() { return; }
    facade::gf_train_save_json(&(*gen).inner, &(*disc).inner, cstr(path));
}

/// Load both networks from a JSON file.
#[no_mangle]
pub unsafe extern "C" fn gf_train_load_json(
    gen: *mut GanNetwork, disc: *mut GanNetwork, path: *const c_char,
) {
    if gen.is_null() || disc.is_null() { return; }
    facade::gf_train_load_json(&mut (*gen).inner, &mut (*disc).inner, cstr(path));
}

/// Save a checkpoint to dir at epoch ep.
#[no_mangle]
pub unsafe extern "C" fn gf_train_save_checkpoint(
    gen: *const GanNetwork, disc: *const GanNetwork, ep: c_int, dir: *const c_char,
) {
    if gen.is_null() || disc.is_null() { return; }
    facade::gf_train_save_checkpoint(&(*gen).inner, &(*disc).inner, ep, cstr(dir));
}

/// Load a checkpoint from dir at epoch ep.
#[no_mangle]
pub unsafe extern "C" fn gf_train_load_checkpoint(
    gen: *mut GanNetwork, disc: *mut GanNetwork, ep: c_int, dir: *const c_char,
) {
    if gen.is_null() || disc.is_null() { return; }
    facade::gf_train_load_checkpoint(&mut (*gen).inner, &mut (*disc).inner, ep, cstr(dir));
}

// =========================================================================
// Metrics API
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn gf_metrics_free(m: *mut GanMetrics) {
    if !m.is_null() { drop(Box::from_raw(m)); }
}
#[no_mangle] pub unsafe extern "C" fn gf_metrics_d_loss_real(m: *const GanMetrics) -> c_float {
    if m.is_null() { 0.0 } else { (*m).inner.d_loss_real }
}
#[no_mangle] pub unsafe extern "C" fn gf_metrics_d_loss_fake(m: *const GanMetrics) -> c_float {
    if m.is_null() { 0.0 } else { (*m).inner.d_loss_fake }
}
#[no_mangle] pub unsafe extern "C" fn gf_metrics_g_loss(m: *const GanMetrics) -> c_float {
    if m.is_null() { 0.0 } else { (*m).inner.g_loss }
}
#[no_mangle] pub unsafe extern "C" fn gf_metrics_fid_score(m: *const GanMetrics) -> c_float {
    if m.is_null() { 0.0 } else { (*m).inner.fid_score }
}
#[no_mangle] pub unsafe extern "C" fn gf_metrics_is_score(m: *const GanMetrics) -> c_float {
    if m.is_null() { 0.0 } else { (*m).inner.is_score }
}
#[no_mangle] pub unsafe extern "C" fn gf_metrics_grad_penalty(m: *const GanMetrics) -> c_float {
    if m.is_null() { 0.0 } else { (*m).inner.grad_penalty }
}
#[no_mangle] pub unsafe extern "C" fn gf_metrics_epoch(m: *const GanMetrics) -> c_int {
    if m.is_null() { 0 } else { (*m).inner.epoch }
}
#[no_mangle] pub unsafe extern "C" fn gf_metrics_batch(m: *const GanMetrics) -> c_int {
    if m.is_null() { 0 } else { (*m).inner.batch }
}

// =========================================================================
// High-level Run API
// =========================================================================

/// Full pipeline: build → train → return result.  Free with gf_result_free().
#[no_mangle]
pub unsafe extern "C" fn gf_run(cfg: *const GanConfig) -> *mut GanResult {
    if cfg.is_null() { return std::ptr::null_mut(); }
    let r = facade::gf_run(&(*cfg).inner);
    Box::into_raw(Box::new(GanResult { inner: r }))
}

#[no_mangle]
pub unsafe extern "C" fn gf_result_free(r: *mut GanResult) {
    if !r.is_null() { drop(Box::from_raw(r)); }
}

/// Returns a NEW GanNetwork* (clone).  Caller must free with gf_network_free().
#[no_mangle]
pub unsafe extern "C" fn gf_result_generator(r: *const GanResult) -> *mut GanNetwork {
    if r.is_null() { return std::ptr::null_mut(); }
    Box::into_raw(Box::new(GanNetwork { inner: (*r).inner.generator.clone() }))
}

/// Returns a NEW GanNetwork* (clone).  Caller must free with gf_network_free().
#[no_mangle]
pub unsafe extern "C" fn gf_result_discriminator(r: *const GanResult) -> *mut GanNetwork {
    if r.is_null() { return std::ptr::null_mut(); }
    Box::into_raw(Box::new(GanNetwork { inner: (*r).inner.discriminator.clone() }))
}

/// Returns a NEW GanMetrics* (clone).  Caller must free with gf_metrics_free().
#[no_mangle]
pub unsafe extern "C" fn gf_result_metrics(r: *const GanResult) -> *mut GanMetrics {
    if r.is_null() { return std::ptr::null_mut(); }
    Box::into_raw(Box::new(GanMetrics { inner: (*r).inner.metrics.clone() }))
}

// =========================================================================
// Loss API
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn gf_bce_loss(pred: *const GanMatrix, target: *const GanMatrix) -> c_float {
    facaded_gan_cuda::loss::binary_cross_entropy(&gan_to_matrix(pred), &gan_to_matrix(target))
}
#[no_mangle]
pub unsafe extern "C" fn gf_bce_grad(pred: *const GanMatrix, target: *const GanMatrix) -> *mut GanMatrix {
    matrix_to_gan(&facaded_gan_cuda::loss::bce_gradient(&gan_to_matrix(pred), &gan_to_matrix(target)))
}
#[no_mangle]
pub unsafe extern "C" fn gf_wgan_disc_loss(d_real: *const GanMatrix, d_fake: *const GanMatrix) -> c_float {
    facaded_gan_cuda::loss::wgan_disc_loss(&gan_to_matrix(d_real), &gan_to_matrix(d_fake))
}
#[no_mangle]
pub unsafe extern "C" fn gf_wgan_gen_loss(d_fake: *const GanMatrix) -> c_float {
    facaded_gan_cuda::loss::wgan_gen_loss(&gan_to_matrix(d_fake))
}
#[no_mangle]
pub unsafe extern "C" fn gf_hinge_disc_loss(d_real: *const GanMatrix, d_fake: *const GanMatrix) -> c_float {
    facaded_gan_cuda::loss::hinge_disc_loss(&gan_to_matrix(d_real), &gan_to_matrix(d_fake))
}
#[no_mangle]
pub unsafe extern "C" fn gf_hinge_gen_loss(d_fake: *const GanMatrix) -> c_float {
    facaded_gan_cuda::loss::hinge_gen_loss(&gan_to_matrix(d_fake))
}
#[no_mangle]
pub unsafe extern "C" fn gf_ls_disc_loss(d_real: *const GanMatrix, d_fake: *const GanMatrix) -> c_float {
    facaded_gan_cuda::loss::ls_disc_loss(&gan_to_matrix(d_real), &gan_to_matrix(d_fake))
}
#[no_mangle]
pub unsafe extern "C" fn gf_ls_gen_loss(d_fake: *const GanMatrix) -> c_float {
    facaded_gan_cuda::loss::ls_gen_loss(&gan_to_matrix(d_fake))
}
#[no_mangle]
pub unsafe extern "C" fn gf_cosine_anneal(
    epoch: c_int, max_ep: c_int, base_lr: c_float, min_lr: c_float,
) -> c_float {
    facaded_gan_cuda::optimizer::cosine_anneal(epoch, max_ep, base_lr, min_lr)
}

// =========================================================================
// Activation API
// =========================================================================

/// Apply a named activation.  act_type: "relu","sigmoid","tanh","leaky","none".
/// Free result with gf_matrix_free().
#[no_mangle]
pub unsafe extern "C" fn gf_activate(a: *const GanMatrix, act_type: *const c_char) -> *mut GanMatrix {
    matrix_to_gan(&facaded_gan_cuda::activations::apply_activation(
        &gan_to_matrix(a), parse_activation(cstr(act_type))
    ))
}
#[no_mangle]
pub unsafe extern "C" fn gf_relu(a: *const GanMatrix) -> *mut GanMatrix {
    matrix_to_gan(&facaded_gan_cuda::activations::matrix_relu(&gan_to_matrix(a)))
}
#[no_mangle]
pub unsafe extern "C" fn gf_sigmoid(a: *const GanMatrix) -> *mut GanMatrix {
    matrix_to_gan(&facaded_gan_cuda::activations::matrix_sigmoid(&gan_to_matrix(a)))
}
#[no_mangle]
pub unsafe extern "C" fn gf_tanh_m(a: *const GanMatrix) -> *mut GanMatrix {
    matrix_to_gan(&facaded_gan_cuda::activations::matrix_tanh(&gan_to_matrix(a)))
}
#[no_mangle]
pub unsafe extern "C" fn gf_leaky_relu(a: *const GanMatrix, alpha: c_float) -> *mut GanMatrix {
    matrix_to_gan(&facaded_gan_cuda::activations::matrix_leaky_relu(&gan_to_matrix(a), alpha))
}
#[no_mangle]
pub unsafe extern "C" fn gf_softmax(a: *const GanMatrix) -> *mut GanMatrix {
    matrix_to_gan(&facaded_gan_cuda::activations::matrix_softmax(&gan_to_matrix(a)))
}

// =========================================================================
// Random / Noise API
// =========================================================================

#[no_mangle]
pub extern "C" fn gf_random_gaussian() -> c_float {
    facaded_gan_cuda::random::random_gaussian()
}
#[no_mangle]
pub extern "C" fn gf_random_uniform(lo: c_float, hi: c_float) -> c_float {
    facaded_gan_cuda::random::random_uniform(lo, hi)
}

/// Generate a size×depth noise matrix.  noise_type: "gauss","uniform","analog".
/// Free with gf_matrix_free().
#[no_mangle]
pub unsafe extern "C" fn gf_generate_noise(
    size: c_int, depth: c_int, noise_type: *const c_char,
) -> *mut GanMatrix {
    let mut v = vec![];
    facaded_gan_cuda::random::generate_noise(&mut v, size, depth, parse_noise_type(cstr(noise_type)));
    matrix_to_gan(&v)
}

// =========================================================================
// Security API
// =========================================================================

/// Returns 1 if path passes validation, 0 otherwise.
#[no_mangle]
pub unsafe extern "C" fn gf_validate_path(path: *const c_char) -> c_int {
    facaded_gan_cuda::security::validate_path(cstr(path)) as c_int
}

/// Append msg to log_file with a timestamp.
#[no_mangle]
pub unsafe extern "C" fn gf_audit_log(msg: *const c_char, log_file: *const c_char) {
    facaded_gan_cuda::security::audit_log(cstr(msg), cstr(log_file));
}

/// Returns 1 if (r,c) is within matrix bounds, 0 otherwise.
#[no_mangle]
pub unsafe extern "C" fn gf_bounds_check(m: *const GanMatrix, r: c_int, c: c_int) -> c_int {
    facaded_gan_cuda::security::bounds_check(&gan_to_matrix(m), r, c) as c_int
}

// =========================================================================
// Backend API
// =========================================================================

/// Initialise the global compute backend.  name: "cpu","cuda","opencl","hybrid","auto".
#[no_mangle]
pub unsafe extern "C" fn gf_init_backend(name: *const c_char) {
    backend::init_backend(parse_backend(cstr(name)));
}

/// Returns a static string naming the best detected backend; do not free.
#[no_mangle]
pub extern "C" fn gf_detect_backend() -> *const c_char {
    match backend::detect_best_backend() {
        ComputeBackend::CPU    => b"cpu\0".as_ptr()    as *const c_char,
        ComputeBackend::CUDA   => b"cuda\0".as_ptr()   as *const c_char,
        ComputeBackend::OpenCL => b"opencl\0".as_ptr() as *const c_char,
        ComputeBackend::Hybrid => b"hybrid\0".as_ptr() as *const c_char,
    }
}

/// Seed the RNG from /dev/urandom.
#[no_mangle]
pub extern "C" fn gf_secure_randomize() {
    facade::gf_sec_secure_randomize();
}

// =========================================================================
// Kani FFI verification harnesses
// =========================================================================

#[cfg(kani)]
mod kani_ffi_tests;

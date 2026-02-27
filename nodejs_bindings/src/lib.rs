/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Node.js bindings for facaded_gan_cuda via napi-rs 2.x.
 *
 * USAGE (after `napi build --platform --release`)
 * ------------------------------------------------
 *   const gan = require('facaded-gan');
 *
 *   const cfg = new gan.GanConfig();
 *   cfg.epochs     = 2;
 *   cfg.batch_size = 8;
 *
 *   gan.init_backend('cpu');
 *   const result = gan.run(cfg);
 *   console.log('g_loss:', result.metrics.g_loss);
 *   console.log('gen layers:', result.generator.layer_count);
 *
 *   // Low-level
 *   const gen = gan.gen_build([64, 128, 1], 'leaky', 'adam', 0.0002);
 *   const noise = gan.generate_noise(8, 64, 'gauss');
 *   const out = gen.forward(noise);
 *   console.log('output rows:', out.rows);
 */

#![allow(clippy::all)]

use napi_derive::napi;

use facaded_gan_cuda::types::*;
use facaded_gan_cuda::facade;
use facaded_gan_cuda::backend::{self, ComputeBackend};

// ─── Enum helpers ─────────────────────────────────────────────────────────────

fn parse_activation(s: &str) -> ActivationType {
    match s {
        "sigmoid" => ActivationType::Sigmoid,
        "tanh"    => ActivationType::Tanh,
        "leaky"   => ActivationType::LeakyReLU,
        "none"    => ActivationType::None,
        _         => ActivationType::ReLU,
    }
}

fn activation_str(a: ActivationType) -> &'static str {
    match a {
        ActivationType::ReLU     => "relu",
        ActivationType::Sigmoid  => "sigmoid",
        ActivationType::Tanh     => "tanh",
        ActivationType::LeakyReLU=> "leaky",
        ActivationType::None     => "none",
    }
}

fn parse_optimizer(s: &str) -> Optimizer {
    match s {
        "sgd"     => Optimizer::SGD,
        "rmsprop" => Optimizer::RMSProp,
        _         => Optimizer::Adam,
    }
}

fn optimizer_str(o: Optimizer) -> &'static str {
    match o {
        Optimizer::Adam    => "adam",
        Optimizer::SGD     => "sgd",
        Optimizer::RMSProp => "rmsprop",
    }
}

fn parse_noise_type(s: &str) -> NoiseType {
    match s {
        "uniform" => NoiseType::Uniform,
        "analog"  => NoiseType::Analog,
        _         => NoiseType::Gauss,
    }
}

fn parse_data_type(s: &str) -> DataType {
    match s {
        "image" => DataType::Image,
        "audio" => DataType::Audio,
        _       => DataType::Vector,
    }
}

fn data_type_str(d: DataType) -> &'static str {
    match d {
        DataType::Vector => "vector",
        DataType::Image  => "image",
        DataType::Audio  => "audio",
    }
}

fn parse_loss_type(s: &str) -> LossType {
    match s {
        "wgan"  => LossType::WGANGP,
        "hinge" => LossType::Hinge,
        "ls"    => LossType::LeastSquares,
        _       => LossType::BCE,
    }
}

fn loss_type_str(l: LossType) -> &'static str {
    match l {
        LossType::BCE          => "bce",
        LossType::WGANGP       => "wgan",
        LossType::Hinge        => "hinge",
        LossType::LeastSquares => "ls",
    }
}

// ─── Matrix ───────────────────────────────────────────────────────────────────

/// A 2-D matrix of f32 values (rows × cols), backed by a Vec<Vec<f32>>.
#[napi]
pub struct Matrix {
    inner: TMatrix,
}

#[napi]
impl Matrix {
    /// Create a zero-filled rows×cols matrix.
    #[napi(constructor)]
    pub fn new(rows: i32, cols: i32) -> Self {
        let r = rows.max(0) as usize;
        let c = cols.max(0) as usize;
        Self { inner: vec![vec![0.0f32; c]; r] }
    }

    /// Build a Matrix from a nested JS number array.
    #[napi(factory)]
    pub fn from_array(data: Vec<Vec<f64>>) -> Self {
        Self {
            inner: data.into_iter()
                .map(|row| row.into_iter().map(|v| v as f32).collect())
                .collect(),
        }
    }

    /// Number of rows.
    #[napi(getter)]
    pub fn rows(&self) -> i32 { self.inner.len() as i32 }

    /// Number of columns (of the first row; 0 if empty).
    #[napi(getter)]
    pub fn cols(&self) -> i32 {
        self.inner.first().map(|r| r.len() as i32).unwrap_or(0)
    }

    /// Element read (returns 0 on out-of-range).
    #[napi]
    pub fn get(&self, row: i32, col: i32) -> f64 {
        self.inner
            .get(row as usize)
            .and_then(|r| r.get(col as usize))
            .copied()
            .unwrap_or(0.0) as f64
    }

    /// Element write (no-op on out-of-range).
    #[napi]
    pub fn set(&mut self, row: i32, col: i32, val: f64) {
        if let Some(r) = self.inner.get_mut(row as usize) {
            if let Some(v) = r.get_mut(col as usize) {
                *v = val as f32;
            }
        }
    }

    /// Safe element read (returns def on out-of-range).
    #[napi]
    pub fn safe_get(&self, row: i32, col: i32, def: f64) -> f64 {
        facade::gf_op_safe_get(&self.inner, row, col, def as f32) as f64
    }

    /// Export all rows as a JS Array<Array<number>>.
    #[napi]
    pub fn to_array(&self) -> Vec<Vec<f64>> {
        self.inner.iter()
            .map(|row| row.iter().map(|&v| v as f64).collect())
            .collect()
    }

    // ── Arithmetic ──────────────────────────────────────────────────────────

    #[napi]
    pub fn multiply(&self, b: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_matrix_multiply(&self.inner, &b.inner) }
    }

    #[napi]
    pub fn add(&self, b: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_matrix_add(&self.inner, &b.inner) }
    }

    #[napi]
    pub fn subtract(&self, b: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_matrix_subtract(&self.inner, &b.inner) }
    }

    #[napi]
    pub fn scale(&self, s: f64) -> Matrix {
        Matrix { inner: facade::gf_op_matrix_scale(&self.inner, s as f32) }
    }

    #[napi]
    pub fn transpose(&self) -> Matrix {
        Matrix { inner: facade::gf_op_matrix_transpose(&self.inner) }
    }

    #[napi]
    pub fn normalize(&self) -> Matrix {
        Matrix { inner: facade::gf_op_matrix_normalize(&self.inner) }
    }

    #[napi]
    pub fn element_mul(&self, b: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_matrix_element_mul(&self.inner, &b.inner) }
    }

    // ── Activations ─────────────────────────────────────────────────────────

    #[napi]
    pub fn relu(&self) -> Matrix {
        Matrix { inner: facade::gf_op_relu(&self.inner) }
    }

    #[napi]
    pub fn sigmoid(&self) -> Matrix {
        Matrix { inner: facade::gf_op_sigmoid(&self.inner) }
    }

    #[napi]
    pub fn tanh_act(&self) -> Matrix {
        Matrix { inner: facade::gf_op_tanh(&self.inner) }
    }

    #[napi]
    pub fn leaky_relu(&self, alpha: f64) -> Matrix {
        Matrix { inner: facade::gf_op_leaky_relu(&self.inner, alpha as f32) }
    }

    #[napi]
    pub fn softmax(&self) -> Matrix {
        Matrix { inner: facade::gf_op_softmax(&self.inner) }
    }

    /// Apply a named activation: "relu" | "sigmoid" | "tanh" | "leaky" | "none".
    #[napi]
    pub fn activate(&self, act: String) -> Matrix {
        Matrix { inner: facade::gf_op_activate(&self.inner, parse_activation(&act)) }
    }
}

// ─── GanConfig ────────────────────────────────────────────────────────────────

/// Training hyper-parameters. Construct with `new GanConfig()`.
#[napi]
pub struct GanConfig {
    inner: GANConfig,
}

#[napi]
impl GanConfig {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self { inner: GANConfig::default() }
    }

    // Integer fields
    #[napi(getter)] pub fn epochs(&self) -> i32 { self.inner.epochs }
    #[napi(setter)] pub fn set_epochs(&mut self, v: i32) { self.inner.epochs = v; }

    #[napi(getter)] pub fn batch_size(&self) -> i32 { self.inner.batch_size }
    #[napi(setter)] pub fn set_batch_size(&mut self, v: i32) { self.inner.batch_size = v; }

    #[napi(getter)] pub fn noise_depth(&self) -> i32 { self.inner.noise_depth }
    #[napi(setter)] pub fn set_noise_depth(&mut self, v: i32) { self.inner.noise_depth = v; }

    #[napi(getter)] pub fn condition_size(&self) -> i32 { self.inner.condition_size }
    #[napi(setter)] pub fn set_condition_size(&mut self, v: i32) { self.inner.condition_size = v; }

    #[napi(getter)] pub fn generator_bits(&self) -> i32 { self.inner.generator_bits }
    #[napi(setter)] pub fn set_generator_bits(&mut self, v: i32) { self.inner.generator_bits = v; }

    #[napi(getter)] pub fn discriminator_bits(&self) -> i32 { self.inner.discriminator_bits }
    #[napi(setter)] pub fn set_discriminator_bits(&mut self, v: i32) { self.inner.discriminator_bits = v; }

    #[napi(getter)] pub fn max_res_level(&self) -> i32 { self.inner.max_res_level }
    #[napi(setter)] pub fn set_max_res_level(&mut self, v: i32) { self.inner.max_res_level = v; }

    #[napi(getter)] pub fn metric_interval(&self) -> i32 { self.inner.metric_interval }
    #[napi(setter)] pub fn set_metric_interval(&mut self, v: i32) { self.inner.metric_interval = v; }

    #[napi(getter)] pub fn checkpoint_interval(&self) -> i32 { self.inner.checkpoint_interval }
    #[napi(setter)] pub fn set_checkpoint_interval(&mut self, v: i32) { self.inner.checkpoint_interval = v; }

    #[napi(getter)] pub fn fuzz_iterations(&self) -> i32 { self.inner.fuzz_iterations }
    #[napi(setter)] pub fn set_fuzz_iterations(&mut self, v: i32) { self.inner.fuzz_iterations = v; }

    #[napi(getter)] pub fn num_threads(&self) -> i32 { self.inner.num_threads }
    #[napi(setter)] pub fn set_num_threads(&mut self, v: i32) { self.inner.num_threads = v; }

    // Float fields
    #[napi(getter)] pub fn learning_rate(&self) -> f64 { self.inner.learning_rate as f64 }
    #[napi(setter)] pub fn set_learning_rate(&mut self, v: f64) { self.inner.learning_rate = v as f32; }

    #[napi(getter)] pub fn gp_lambda(&self) -> f64 { self.inner.gp_lambda as f64 }
    #[napi(setter)] pub fn set_gp_lambda(&mut self, v: f64) { self.inner.gp_lambda = v as f32; }

    #[napi(getter)] pub fn generator_lr(&self) -> f64 { self.inner.generator_lr as f64 }
    #[napi(setter)] pub fn set_generator_lr(&mut self, v: f64) { self.inner.generator_lr = v as f32; }

    #[napi(getter)] pub fn discriminator_lr(&self) -> f64 { self.inner.discriminator_lr as f64 }
    #[napi(setter)] pub fn set_discriminator_lr(&mut self, v: f64) { self.inner.discriminator_lr = v as f32; }

    #[napi(getter)] pub fn weight_decay_val(&self) -> f64 { self.inner.weight_decay_val as f64 }
    #[napi(setter)] pub fn set_weight_decay_val(&mut self, v: f64) { self.inner.weight_decay_val = v as f32; }

    // Bool fields
    #[napi(getter)] pub fn use_batch_norm(&self) -> bool { self.inner.use_batch_norm }
    #[napi(setter)] pub fn set_use_batch_norm(&mut self, v: bool) { self.inner.use_batch_norm = v; }

    #[napi(getter)] pub fn use_layer_norm(&self) -> bool { self.inner.use_layer_norm }
    #[napi(setter)] pub fn set_use_layer_norm(&mut self, v: bool) { self.inner.use_layer_norm = v; }

    #[napi(getter)] pub fn use_spectral_norm(&self) -> bool { self.inner.use_spectral_norm }
    #[napi(setter)] pub fn set_use_spectral_norm(&mut self, v: bool) { self.inner.use_spectral_norm = v; }

    #[napi(getter)] pub fn use_label_smoothing(&self) -> bool { self.inner.use_label_smoothing }
    #[napi(setter)] pub fn set_use_label_smoothing(&mut self, v: bool) { self.inner.use_label_smoothing = v; }

    #[napi(getter)] pub fn use_feature_matching(&self) -> bool { self.inner.use_feature_matching }
    #[napi(setter)] pub fn set_use_feature_matching(&mut self, v: bool) { self.inner.use_feature_matching = v; }

    #[napi(getter)] pub fn use_minibatch_std_dev(&self) -> bool { self.inner.use_minibatch_std_dev }
    #[napi(setter)] pub fn set_use_minibatch_std_dev(&mut self, v: bool) { self.inner.use_minibatch_std_dev = v; }

    #[napi(getter)] pub fn use_progressive(&self) -> bool { self.inner.use_progressive }
    #[napi(setter)] pub fn set_use_progressive(&mut self, v: bool) { self.inner.use_progressive = v; }

    #[napi(getter)] pub fn use_augmentation(&self) -> bool { self.inner.use_augmentation }
    #[napi(setter)] pub fn set_use_augmentation(&mut self, v: bool) { self.inner.use_augmentation = v; }

    #[napi(getter)] pub fn compute_metrics(&self) -> bool { self.inner.compute_metrics }
    #[napi(setter)] pub fn set_compute_metrics(&mut self, v: bool) { self.inner.compute_metrics = v; }

    #[napi(getter)] pub fn use_weight_decay(&self) -> bool { self.inner.use_weight_decay }
    #[napi(setter)] pub fn set_use_weight_decay(&mut self, v: bool) { self.inner.use_weight_decay = v; }

    #[napi(getter)] pub fn use_cosine_anneal(&self) -> bool { self.inner.use_cosine_anneal }
    #[napi(setter)] pub fn set_use_cosine_anneal(&mut self, v: bool) { self.inner.use_cosine_anneal = v; }

    #[napi(getter)] pub fn audit_log(&self) -> bool { self.inner.audit_log }
    #[napi(setter)] pub fn set_audit_log(&mut self, v: bool) { self.inner.audit_log = v; }

    #[napi(getter)] pub fn use_encryption(&self) -> bool { self.inner.use_encryption }
    #[napi(setter)] pub fn set_use_encryption(&mut self, v: bool) { self.inner.use_encryption = v; }

    #[napi(getter)] pub fn use_conv(&self) -> bool { self.inner.use_conv }
    #[napi(setter)] pub fn set_use_conv(&mut self, v: bool) { self.inner.use_conv = v; }

    #[napi(getter)] pub fn use_attention(&self) -> bool { self.inner.use_attention }
    #[napi(setter)] pub fn set_use_attention(&mut self, v: bool) { self.inner.use_attention = v; }

    // String fields — getters + setters
    #[napi(getter)] pub fn save_model(&self) -> String { self.inner.save_model.clone() }
    #[napi(setter)] pub fn set_save_model(&mut self, v: String) { self.inner.save_model = v; }

    #[napi(getter)] pub fn load_model(&self) -> String { self.inner.load_model.clone() }
    #[napi(setter)] pub fn set_load_model(&mut self, v: String) { self.inner.load_model = v; }

    #[napi(getter)] pub fn load_json_model(&self) -> String { self.inner.load_json_model.clone() }
    #[napi(setter)] pub fn set_load_json_model(&mut self, v: String) { self.inner.load_json_model = v; }

    #[napi(getter)] pub fn output_dir(&self) -> String { self.inner.output_dir.clone() }
    #[napi(setter)] pub fn set_output_dir(&mut self, v: String) { self.inner.output_dir = v; }

    #[napi(getter)] pub fn data_path(&self) -> String { self.inner.data_path.clone() }
    #[napi(setter)] pub fn set_data_path(&mut self, v: String) { self.inner.data_path = v; }

    #[napi(getter)] pub fn audit_log_file(&self) -> String { self.inner.audit_log_file.clone() }
    #[napi(setter)] pub fn set_audit_log_file(&mut self, v: String) { self.inner.audit_log_file = v; }

    #[napi(getter)] pub fn encryption_key(&self) -> String { self.inner.encryption_key.clone() }
    #[napi(setter)] pub fn set_encryption_key(&mut self, v: String) { self.inner.encryption_key = v; }

    // Enum fields — string in/out
    #[napi(getter)] pub fn activation(&self) -> String { activation_str(self.inner.activation).to_string() }
    #[napi(setter)] pub fn set_activation(&mut self, v: String) { self.inner.activation = parse_activation(&v); }

    #[napi(getter)] pub fn optimizer(&self) -> String { optimizer_str(self.inner.optimizer).to_string() }
    #[napi(setter)] pub fn set_optimizer(&mut self, v: String) { self.inner.optimizer = parse_optimizer(&v); }

    #[napi(getter)] pub fn loss_type(&self) -> String { loss_type_str(self.inner.loss_type).to_string() }
    #[napi(setter)] pub fn set_loss_type(&mut self, v: String) { self.inner.loss_type = parse_loss_type(&v); }

    #[napi(getter)] pub fn noise_type(&self) -> String {
        match self.inner.noise_type {
            NoiseType::Gauss   => "gauss",
            NoiseType::Uniform => "uniform",
            NoiseType::Analog  => "analog",
        }.to_string()
    }
    #[napi(setter)] pub fn set_noise_type(&mut self, v: String) { self.inner.noise_type = parse_noise_type(&v); }

    #[napi(getter)] pub fn data_type(&self) -> String { data_type_str(self.inner.data_type).to_string() }
    #[napi(setter)] pub fn set_data_type(&mut self, v: String) { self.inner.data_type = parse_data_type(&v); }
}

// ─── GanNetwork ───────────────────────────────────────────────────────────────

/// A generator or discriminator network.
/// Obtain via `gen_build()`, `disc_build()`, `gen_build_conv()`, `disc_build_conv()`, or `GanResult.generator`.
#[napi]
pub struct GanNetwork {
    inner: Network,
}

#[napi]
impl GanNetwork {
    #[napi(getter)] pub fn layer_count(&self) -> i32 { self.inner.layer_count }
    #[napi(getter)] pub fn learning_rate(&self) -> f64 { self.inner.learning_rate as f64 }
    #[napi(getter)] pub fn is_training(&self) -> bool { self.inner.is_training }
    #[napi(getter)] pub fn optimizer(&self) -> String { optimizer_str(self.inner.optimizer).to_string() }

    /// Run a forward pass. `inp` is batch×features.
    #[napi]
    pub fn forward(&mut self, inp: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_gen_forward(&mut self.inner, &inp.inner) }
    }

    /// Run a backward pass.
    #[napi]
    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_gen_backward(&mut self.inner, &grad_out.inner) }
    }

    /// Apply accumulated gradients to weights.
    #[napi]
    pub fn update_weights(&mut self) {
        facade::gf_gen_update_weights(&mut self.inner);
    }

    /// Switch training (true) / inference (false) mode.
    #[napi]
    pub fn set_training(&mut self, training: bool) {
        facade::gf_gen_set_training(&mut self.inner, training);
    }

    /// Sample count outputs. noise_type: "gauss" | "uniform" | "analog".
    #[napi]
    pub fn sample(&mut self, count: i32, noise_dim: i32, noise_type: String) -> Matrix {
        Matrix {
            inner: facade::gf_gen_sample(
                &mut self.inner, count, noise_dim,
                parse_noise_type(&noise_type),
            ),
        }
    }

    /// Sanitise weights (replace NaN/Inf with 0).
    #[napi]
    pub fn verify(&mut self) {
        facade::gf_sec_verify_network(&mut self.inner);
    }

    /// Save weights to path (binary).
    #[napi]
    pub fn save(&self, path: String) {
        facade::gf_train_save_model(&self.inner, &path);
    }

    /// Load weights from path (binary).
    #[napi]
    pub fn load(&mut self, path: String) {
        facade::gf_train_load_model(&mut self.inner, &path);
    }

    /// Return a deep copy of this network.
    #[napi]
    pub fn deep_copy(&self) -> GanNetwork {
        GanNetwork { inner: facade::gf_gen_deep_copy(&self.inner) }
    }
}

// ─── GanDataset ───────────────────────────────────────────────────────────────

/// A training dataset.
/// Obtain via `GanDataset.synthetic()` or `GanDataset.load()`.
#[napi]
pub struct GanDataset {
    inner: Dataset,
}

#[napi]
impl GanDataset {
    /// Create a synthetic random dataset.
    #[napi(factory)]
    pub fn synthetic(count: i32, features: i32) -> Self {
        Self { inner: facade::gf_train_create_synthetic(count, features) }
    }

    /// Load a dataset from path. data_type: "vector" | "image" | "audio".
    #[napi(factory)]
    pub fn load(path: String, data_type: String) -> Self {
        Self { inner: facade::gf_train_load_dataset(&path, parse_data_type(&data_type)) }
    }

    /// Number of samples.
    #[napi(getter)]
    pub fn count(&self) -> i32 { self.inner.count }

    /// Data type: "vector" | "image" | "audio".
    #[napi(getter)]
    pub fn data_type(&self) -> String { data_type_str(self.inner.data_type).to_string() }
}

// ─── GanMetrics ───────────────────────────────────────────────────────────────

/// Per-step training statistics.
#[napi]
pub struct GanMetrics {
    inner: GANMetrics,
}

#[napi]
impl GanMetrics {
    #[napi(getter)] pub fn d_loss_real(&self) -> f64  { self.inner.d_loss_real as f64 }
    #[napi(getter)] pub fn d_loss_fake(&self) -> f64  { self.inner.d_loss_fake as f64 }
    #[napi(getter)] pub fn g_loss(&self) -> f64       { self.inner.g_loss as f64 }
    #[napi(getter)] pub fn fid_score(&self) -> f64    { self.inner.fid_score as f64 }
    #[napi(getter)] pub fn is_score(&self) -> f64     { self.inner.is_score as f64 }
    #[napi(getter)] pub fn grad_penalty(&self) -> f64 { self.inner.grad_penalty as f64 }
    #[napi(getter)] pub fn epoch(&self) -> i32        { self.inner.epoch }
    #[napi(getter)] pub fn batch(&self) -> i32        { self.inner.batch }
}

// ─── GanResult ────────────────────────────────────────────────────────────────

/// Combined result of `run()`: trained networks + final metrics.
#[napi]
pub struct GanResult {
    inner: GANResult,
}

#[napi]
impl GanResult {
    /// Cloned trained generator network.
    #[napi(getter)]
    pub fn generator(&self) -> GanNetwork {
        GanNetwork { inner: self.inner.generator.clone() }
    }

    /// Cloned trained discriminator network.
    #[napi(getter)]
    pub fn discriminator(&self) -> GanNetwork {
        GanNetwork { inner: self.inner.discriminator.clone() }
    }

    /// Final training metrics.
    #[napi(getter)]
    pub fn metrics(&self) -> GanMetrics {
        GanMetrics { inner: self.inner.metrics.clone() }
    }
}

// ─── High-level API ───────────────────────────────────────────────────────────

/// Build networks, train, and return results — all in one call.
#[napi]
pub fn run(config: &GanConfig) -> GanResult {
    GanResult { inner: facade::gf_run(&config.inner) }
}

/// Initialise the global compute backend.
/// name: "cpu" | "cuda" | "opencl" | "hybrid" | "auto"
#[napi]
pub fn init_backend(name: String) {
    let be = match name.as_str() {
        "cuda"   => ComputeBackend::CUDA,
        "opencl" => ComputeBackend::OpenCL,
        "hybrid" => ComputeBackend::Hybrid,
        _        => ComputeBackend::CPU,
    };
    backend::init_backend(be);
}

/// Detect the best available backend. Returns "cpu", "cuda", "opencl", or "hybrid".
#[napi]
pub fn detect_backend() -> String {
    format!("{}", backend::detect_best_backend()).to_lowercase()
}

// ─── Network factory functions ─────────────────────────────────────────────────

/// Build a dense generator. sizes: layer widths, e.g. [64, 128, 1].
#[napi]
pub fn gen_build(sizes: Vec<i32>, act: String, opt: String, lr: f64) -> GanNetwork {
    GanNetwork {
        inner: facade::gf_gen_build(&sizes, parse_activation(&act), parse_optimizer(&opt), lr as f32),
    }
}

/// Build a convolutional generator.
#[napi]
pub fn gen_build_conv(
    noise_dim: i32,
    cond_sz: i32,
    base_ch: i32,
    act: String,
    opt: String,
    lr: f64,
) -> GanNetwork {
    GanNetwork {
        inner: facade::gf_gen_build_conv(
            noise_dim, cond_sz, base_ch,
            parse_activation(&act), parse_optimizer(&opt), lr as f32,
        ),
    }
}

/// Build a dense discriminator. sizes: layer widths.
#[napi]
pub fn disc_build(sizes: Vec<i32>, act: String, opt: String, lr: f64) -> GanNetwork {
    GanNetwork {
        inner: facade::gf_disc_build(&sizes, parse_activation(&act), parse_optimizer(&opt), lr as f32),
    }
}

/// Build a convolutional discriminator.
#[napi]
pub fn disc_build_conv(
    in_ch: i32,
    in_w: i32,
    in_h: i32,
    cond_sz: i32,
    base_ch: i32,
    act: String,
    opt: String,
    lr: f64,
) -> GanNetwork {
    GanNetwork {
        inner: facade::gf_disc_build_conv(
            in_ch, in_w, in_h, cond_sz, base_ch,
            parse_activation(&act), parse_optimizer(&opt), lr as f32,
        ),
    }
}

// ─── Training functions ────────────────────────────────────────────────────────

/// Run all epochs. Returns final-step metrics.
#[napi]
pub fn train_full(
    gen: &mut GanNetwork,
    disc: &mut GanNetwork,
    ds: &GanDataset,
    cfg: &GanConfig,
) -> GanMetrics {
    GanMetrics {
        inner: facade::gf_train_full(&mut gen.inner, &mut disc.inner, &ds.inner, &cfg.inner),
    }
}

/// Run one discriminator + generator update step.
#[napi]
pub fn train_step(
    gen: &mut GanNetwork,
    disc: &mut GanNetwork,
    real_batch: &Matrix,
    noise: &Matrix,
    cfg: &GanConfig,
) -> GanMetrics {
    GanMetrics {
        inner: facade::gf_train_step(
            &mut gen.inner, &mut disc.inner,
            &real_batch.inner, &noise.inner,
            &cfg.inner,
        ),
    }
}

/// Save both networks to a JSON file.
#[napi]
pub fn save_json(gen: &GanNetwork, disc: &GanNetwork, path: String) {
    facade::gf_train_save_json(&gen.inner, &disc.inner, &path);
}

/// Load both networks from a JSON file.
#[napi]
pub fn load_json(gen: &mut GanNetwork, disc: &mut GanNetwork, path: String) {
    facade::gf_train_load_json(&mut gen.inner, &mut disc.inner, &path);
}

/// Save a checkpoint (binary) to dir at epoch ep.
#[napi]
pub fn save_checkpoint(gen: &GanNetwork, disc: &GanNetwork, ep: i32, dir: String) {
    facade::gf_train_save_checkpoint(&gen.inner, &disc.inner, ep, &dir);
}

/// Load a checkpoint from dir at epoch ep.
#[napi]
pub fn load_checkpoint(gen: &mut GanNetwork, disc: &mut GanNetwork, ep: i32, dir: String) {
    facade::gf_train_load_checkpoint(&mut gen.inner, &mut disc.inner, ep, &dir);
}

// ─── Loss functions ────────────────────────────────────────────────────────────

#[napi]
pub fn bce_loss(pred: &Matrix, target: &Matrix) -> f64 {
    facade::gf_train_bce_loss(&pred.inner, &target.inner) as f64
}

#[napi]
pub fn bce_grad(pred: &Matrix, target: &Matrix) -> Matrix {
    Matrix { inner: facade::gf_train_bce_grad(&pred.inner, &target.inner) }
}

#[napi]
pub fn wgan_disc_loss(d_real: &Matrix, d_fake: &Matrix) -> f64 {
    facade::gf_train_wgan_disc_loss(&d_real.inner, &d_fake.inner) as f64
}

#[napi]
pub fn wgan_gen_loss(d_fake: &Matrix) -> f64 {
    facade::gf_train_wgan_gen_loss(&d_fake.inner) as f64
}

#[napi]
pub fn hinge_disc_loss(d_real: &Matrix, d_fake: &Matrix) -> f64 {
    facade::gf_train_hinge_disc_loss(&d_real.inner, &d_fake.inner) as f64
}

#[napi]
pub fn hinge_gen_loss(d_fake: &Matrix) -> f64 {
    facade::gf_train_hinge_gen_loss(&d_fake.inner) as f64
}

#[napi]
pub fn ls_disc_loss(d_real: &Matrix, d_fake: &Matrix) -> f64 {
    facade::gf_train_ls_disc_loss(&d_real.inner, &d_fake.inner) as f64
}

#[napi]
pub fn ls_gen_loss(d_fake: &Matrix) -> f64 {
    facade::gf_train_ls_gen_loss(&d_fake.inner) as f64
}

/// Cosine annealing LR schedule.
#[napi]
pub fn cosine_anneal(epoch: i32, max_ep: i32, base_lr: f64, min_lr: f64) -> f64 {
    facade::gf_train_cosine_anneal(epoch, max_ep, base_lr as f32, min_lr as f32) as f64
}

// ─── Random / Noise ───────────────────────────────────────────────────────────

#[napi]
pub fn random_gaussian() -> f64 {
    facade::gf_op_random_gaussian() as f64
}

#[napi]
pub fn random_uniform(lo: f64, hi: f64) -> f64 {
    facade::gf_op_random_uniform(lo as f32, hi as f32) as f64
}

/// Generate a size×depth noise matrix. noise_type: "gauss" | "uniform" | "analog".
#[napi]
pub fn generate_noise(size: i32, depth: i32, noise_type: String) -> Matrix {
    let mut m: TMatrix = vec![vec![0.0f32; depth as usize]; size as usize];
    facade::gf_op_generate_noise(&mut m, size, depth, parse_noise_type(&noise_type));
    Matrix { inner: m }
}

// ─── Security ─────────────────────────────────────────────────────────────────

/// Returns true if path is safe (no traversal, etc.).
#[napi]
pub fn validate_path(path: String) -> bool {
    facade::gf_sec_validate_path(&path)
}

/// Append msg to log_file with an ISO-8601 timestamp.
#[napi]
pub fn audit_log(msg: String, log_file: String) {
    facade::gf_sec_audit_log(&msg, &log_file);
}

/// Returns true if (r, c) is within matrix bounds.
#[napi]
pub fn bounds_check(m: &Matrix, r: i32, c: i32) -> bool {
    facade::gf_sec_bounds_check(&m.inner, r, c)
}

/// Seed the global RNG from /dev/urandom.
#[napi]
pub fn secure_randomize() {
    facade::gf_sec_secure_randomize();
}

// ─── Matrix in-place operations ───────────────────────────────────────────────

/// Add matrix b into a in place (a += b).
#[napi]
pub fn matrix_add_in_place(a: &mut Matrix, b: &Matrix) {
    facade::gf_op_matrix_add_in_place(&mut a.inner, &b.inner);
}

/// Scale matrix a in place (a *= s).
#[napi]
pub fn matrix_scale_in_place(a: &mut Matrix, s: f64) {
    facade::gf_op_matrix_scale_in_place(&mut a.inner, s as f32);
}

/// Clip all elements of a in place to [lo, hi].
#[napi]
pub fn matrix_clip_in_place(a: &mut Matrix, lo: f64, hi: f64) {
    facade::gf_op_matrix_clip_in_place(&mut a.inner, lo as f32, hi as f32);
}

/// Safe element write (no-op on out-of-range).
#[napi]
pub fn matrix_safe_set(m: &mut Matrix, r: i32, c: i32, val: f64) {
    facade::gf_op_safe_set(&mut m.inner, r, c, val as f32);
}

/// Compute the activation backward pass.
/// act: "relu" | "sigmoid" | "tanh" | "leaky" | "none".
#[napi]
pub fn activation_backward(grad_out: &Matrix, pre_act: &Matrix, act: String) -> Matrix {
    Matrix {
        inner: facade::gf_op_activation_backward(
            &grad_out.inner,
            &pre_act.inner,
            parse_activation(&act),
        ),
    }
}

// ─── Additional GanNetwork methods ────────────────────────────────────────────

#[napi]
impl GanNetwork {
    /// Sample count outputs with a conditioning matrix.
    /// noise_type: "gauss" | "uniform" | "analog".
    #[napi]
    pub fn sample_conditional(
        &mut self,
        count: i32,
        noise_dim: i32,
        cond_sz: i32,
        noise_type: String,
        cond: &Matrix,
    ) -> Matrix {
        Matrix {
            inner: facade::gf_gen_sample_conditional(
                &mut self.inner,
                count,
                noise_dim,
                cond_sz,
                parse_noise_type(&noise_type),
                &cond.inner,
            ),
        }
    }

    /// Add a progressive growing layer at the given resolution level.
    #[napi]
    pub fn gen_add_progressive_layer(&mut self, res_lvl: i32) {
        facade::gf_gen_add_progressive_layer(&mut self.inner, res_lvl);
    }

    /// Get the cached output of layer at index idx.
    #[napi]
    pub fn gen_get_layer_output(&self, idx: i32) -> Matrix {
        Matrix { inner: facade::gf_gen_get_layer_output(&self.inner, idx) }
    }

    /// Return a deep copy of this network (generator variant).
    #[napi]
    pub fn gen_deep_copy(&self) -> GanNetwork {
        GanNetwork { inner: facade::gf_gen_deep_copy(&self.inner) }
    }

    /// Evaluate the discriminator on inp (alias for forward).
    #[napi]
    pub fn disc_evaluate(&mut self, inp: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_disc_evaluate(&mut self.inner, &inp.inner) }
    }

    /// Compute the gradient penalty (WGAN-GP) between real and fake batches.
    #[napi]
    pub fn disc_grad_penalty(&mut self, real: &Matrix, fake: &Matrix, lambda: f64) -> f64 {
        facade::gf_disc_grad_penalty(&mut self.inner, &real.inner, &fake.inner, lambda as f32) as f64
    }

    /// Compute feature-matching loss between real and fake batches at feat_layer.
    #[napi]
    pub fn disc_feature_match(&mut self, real: &Matrix, fake: &Matrix, feat_layer: i32) -> f64 {
        facade::gf_disc_feature_match(&mut self.inner, &real.inner, &fake.inner, feat_layer) as f64
    }

    /// Add a progressive growing layer at the given resolution level (discriminator variant).
    #[napi]
    pub fn disc_add_progressive_layer(&mut self, res_lvl: i32) {
        facade::gf_disc_add_progressive_layer(&mut self.inner, res_lvl);
    }

    /// Get the cached output of discriminator layer at index idx.
    #[napi]
    pub fn disc_get_layer_output(&self, idx: i32) -> Matrix {
        Matrix { inner: facade::gf_disc_get_layer_output(&self.inner, idx) }
    }

    /// Return a deep copy of this network (discriminator variant).
    #[napi]
    pub fn disc_deep_copy(&self) -> GanNetwork {
        GanNetwork { inner: facade::gf_disc_deep_copy(&self.inner) }
    }
}

// ─── Minibatch std dev (top-level) ────────────────────────────────────────────

/// Append a minibatch standard deviation feature column to inp.
#[napi]
pub fn disc_minibatch_std_dev(inp: &Matrix) -> Matrix {
    Matrix { inner: facade::gf_disc_minibatch_std_dev(&inp.inner) }
}

// ─── Training utility functions ───────────────────────────────────────────────

/// Apply accumulated gradients to all layers of net (alias for update_weights).
#[napi]
pub fn train_optimize(net: &mut GanNetwork) {
    facade::gf_train_optimize(&mut net.inner);
}

/// Adam parameter update in place.
#[napi]
pub fn train_adam_update(
    p: &mut Matrix,
    g: &Matrix,
    m_buf: &mut Matrix,
    v_buf: &mut Matrix,
    t: i32,
    lr: f64,
    b1: f64,
    b2: f64,
    eps: f64,
    wd: f64,
) {
    facade::gf_train_adam_update(
        &mut p.inner,
        &g.inner,
        &mut m_buf.inner,
        &mut v_buf.inner,
        t,
        lr as f32,
        b1 as f32,
        b2 as f32,
        eps as f32,
        wd as f32,
    );
}

/// SGD parameter update in place.
#[napi]
pub fn train_sgd_update(p: &mut Matrix, g: &Matrix, lr: f64, wd: f64) {
    facade::gf_train_sgd_update(&mut p.inner, &g.inner, lr as f32, wd as f32);
}

/// RMSProp parameter update in place.
#[napi]
pub fn train_rmsprop_update(
    p: &mut Matrix,
    g: &Matrix,
    cache: &mut Matrix,
    lr: f64,
    decay: f64,
    eps: f64,
    wd: f64,
) {
    facade::gf_train_rmsprop_update(
        &mut p.inner,
        &g.inner,
        &mut cache.inner,
        lr as f32,
        decay as f32,
        eps as f32,
        wd as f32,
    );
}

/// Apply label smoothing to a labels matrix, clamping values into [lo, hi].
#[napi]
pub fn train_label_smoothing(labels: &Matrix, lo: f64, hi: f64) -> Matrix {
    Matrix { inner: facade::gf_train_label_smoothing(&labels.inner, lo as f32, hi as f32) }
}

/// Load a BMP image dataset from path.
#[napi]
pub fn train_load_bmp(path: String) -> GanDataset {
    GanDataset { inner: facade::gf_train_load_bmp(&path) }
}

/// Load a WAV audio dataset from path.
#[napi]
pub fn train_load_wav(path: String) -> GanDataset {
    GanDataset { inner: facade::gf_train_load_wav(&path) }
}

/// Apply data augmentation to sample. data_type: "vector" | "image" | "audio".
#[napi]
pub fn train_augment(sample: &Matrix, data_type: String) -> Matrix {
    Matrix { inner: facade::gf_train_augment(&sample.inner, parse_data_type(&data_type)) }
}

/// Append training metrics to a CSV log file.
#[napi]
pub fn train_log_metrics(m: &GanMetrics, filename: String) {
    facade::gf_train_log_metrics(&m.inner, &filename);
}

/// Save generated samples from gen at epoch ep to dir.
/// noise_type: "gauss" | "uniform" | "analog".
#[napi]
pub fn train_save_samples(
    gen: &mut GanNetwork,
    ep: i32,
    dir: String,
    noise_dim: i32,
    noise_type: String,
) {
    facade::gf_train_save_samples(&mut gen.inner, ep, &dir, noise_dim, parse_noise_type(&noise_type));
}

/// Write discriminator and generator loss curves to a CSV file.
#[napi]
pub fn train_plot_csv(filename: String, d_loss: Vec<f64>, g_loss: Vec<f64>) {
    let d: Vec<f32> = d_loss.iter().map(|&v| v as f32).collect();
    let g: Vec<f32> = g_loss.iter().map(|&v| v as f32).collect();
    let cnt = d.len().min(g.len()) as i32;
    facade::gf_train_plot_csv(&filename, &d, &g, cnt);
}

/// Print an ASCII loss bar to stdout.
#[napi]
pub fn train_print_bar(d_loss: f64, g_loss: f64, width: i32) {
    facade::gf_train_print_bar(d_loss as f32, g_loss as f32, width);
}

/// Compute the Fréchet Inception Distance between real and fake sample arrays.
#[napi]
pub fn train_compute_fid(real_arr: &GanMatrixArray, fake_arr: &GanMatrixArray) -> f64 {
    facade::gf_train_compute_fid(&real_arr.inner, &fake_arr.inner) as f64
}

/// Compute the Inception Score of a sample array.
#[napi]
pub fn train_compute_is(samples: &GanMatrixArray) -> f64 {
    facade::gf_train_compute_is(&samples.inner) as f64
}

// ─── Security functions ────────────────────────────────────────────────────────

/// Return one cryptographically random byte from the OS.
#[napi]
pub fn sec_get_os_random() -> i32 {
    facade::gf_sec_get_os_random() as i32
}

/// Encrypt a model file at in_f, writing ciphertext to out_f using key.
#[napi]
pub fn sec_encrypt_model(in_f: String, out_f: String, key: String) {
    facade::gf_sec_encrypt_model(&in_f, &out_f, &key);
}

/// Decrypt a model file at in_f, writing plaintext to out_f using key.
#[napi]
pub fn sec_decrypt_model(in_f: String, out_f: String, key: String) {
    facade::gf_sec_decrypt_model(&in_f, &out_f, &key);
}

/// Run the built-in security self-tests. Returns 0 on success, 1 on failure.
#[napi]
pub fn sec_run_tests() -> i32 {
    if facade::gf_sec_run_tests() { 0 } else { 1 }
}

/// Run fuzz tests for iterations iterations. Returns 0 on success, 1 on failure.
#[napi]
pub fn sec_run_fuzz_tests(iterations: i32) -> i32 {
    if facade::gf_sec_run_fuzz_tests(iterations) { 0 } else { 1 }
}

// ─── GanLayer ─────────────────────────────────────────────────────────────────

/// A single network layer (dense, conv, norm, attention, etc.).
/// Obtain via the factory functions below.
#[napi]
pub struct GanLayer {
    inner: facaded_gan_cuda::types::Layer,
}

#[napi]
impl GanLayer {
    // ── Factory constructors ─────────────────────────────────────────────────

    /// Create a dense (fully-connected) layer.
    /// act: "relu" | "sigmoid" | "tanh" | "leaky" | "none".
    #[napi(factory)]
    pub fn create_dense(in_sz: i32, out_sz: i32, act: String) -> Self {
        Self { inner: facade::gf_op_create_dense_layer(in_sz, out_sz, parse_activation(&act)) }
    }

    /// Create a Conv2D layer.
    #[napi(factory)]
    pub fn create_conv2d(
        in_ch: i32,
        out_ch: i32,
        k_sz: i32,
        stride: i32,
        pad: i32,
        w: i32,
        h: i32,
        act: String,
    ) -> Self {
        Self {
            inner: facade::gf_op_create_conv2d_layer(
                in_ch, out_ch, k_sz, stride, pad, w, h,
                parse_activation(&act),
            ),
        }
    }

    /// Create a Deconv2D (transposed convolution) layer.
    #[napi(factory)]
    pub fn create_deconv2d(
        in_ch: i32,
        out_ch: i32,
        k_sz: i32,
        stride: i32,
        pad: i32,
        w: i32,
        h: i32,
        act: String,
    ) -> Self {
        Self {
            inner: facade::gf_op_create_deconv2d_layer(
                in_ch, out_ch, k_sz, stride, pad, w, h,
                parse_activation(&act),
            ),
        }
    }

    /// Create a Conv1D layer.
    #[napi(factory)]
    pub fn create_conv1d(
        in_ch: i32,
        out_ch: i32,
        k_sz: i32,
        stride: i32,
        pad: i32,
        in_len: i32,
        act: String,
    ) -> Self {
        Self {
            inner: facade::gf_op_create_conv1d_layer(
                in_ch, out_ch, k_sz, stride, pad, in_len,
                parse_activation(&act),
            ),
        }
    }

    /// Create a batch-normalisation layer.
    #[napi(factory)]
    pub fn create_batch_norm(features: i32) -> Self {
        Self { inner: facade::gf_op_create_batch_norm_layer(features) }
    }

    /// Create a layer-normalisation layer.
    #[napi(factory)]
    pub fn create_layer_norm(features: i32) -> Self {
        Self { inner: facade::gf_op_create_layer_norm_layer(features) }
    }

    /// Create a self-attention layer.
    #[napi(factory)]
    pub fn create_attention(d_model: i32, n_heads: i32) -> Self {
        Self { inner: facade::gf_op_create_attention_layer(d_model, n_heads) }
    }

    // ── Methods ──────────────────────────────────────────────────────────────

    /// Run a forward pass through this layer.
    #[napi]
    pub fn forward(&mut self, inp: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_layer_forward(&mut self.inner, &inp.inner) }
    }

    /// Run a backward pass through this layer.
    #[napi]
    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_layer_backward(&mut self.inner, &grad_out.inner) }
    }

    /// Initialise the optimizer state for this layer.
    /// opt: "adam" | "sgd" | "rmsprop".
    #[napi]
    pub fn init_optimizer(&mut self, opt: String) {
        facade::gf_op_init_layer_optimizer(&mut self.inner, parse_optimizer(&opt));
    }

    /// Run Conv2D forward pass.
    #[napi]
    pub fn conv2d(&mut self, inp: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_conv2d(&inp.inner, &mut self.inner) }
    }

    /// Run Conv2D backward pass.
    #[napi]
    pub fn conv2d_backward(&mut self, grad_out: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_conv2d_backward(&mut self.inner, &grad_out.inner) }
    }

    /// Run Deconv2D forward pass.
    #[napi]
    pub fn deconv2d(&mut self, inp: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_deconv2d(&inp.inner, &mut self.inner) }
    }

    /// Run Deconv2D backward pass.
    #[napi]
    pub fn deconv2d_backward(&mut self, grad_out: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_deconv2d_backward(&mut self.inner, &grad_out.inner) }
    }

    /// Run Conv1D forward pass.
    #[napi]
    pub fn conv1d(&mut self, inp: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_conv1d(&inp.inner, &mut self.inner) }
    }

    /// Run Conv1D backward pass.
    #[napi]
    pub fn conv1d_backward(&mut self, grad_out: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_conv1d_backward(&mut self.inner, &grad_out.inner) }
    }

    /// Run batch-normalisation forward pass.
    #[napi]
    pub fn batch_norm(&mut self, inp: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_batch_norm(&inp.inner, &mut self.inner) }
    }

    /// Run batch-normalisation backward pass.
    #[napi]
    pub fn batch_norm_backward(&mut self, grad_out: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_batch_norm_backward(&mut self.inner, &grad_out.inner) }
    }

    /// Run layer-normalisation forward pass.
    #[napi]
    pub fn layer_norm(&mut self, inp: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_layer_norm(&inp.inner, &mut self.inner) }
    }

    /// Run layer-normalisation backward pass.
    #[napi]
    pub fn layer_norm_backward(&mut self, grad_out: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_layer_norm_backward(&mut self.inner, &grad_out.inner) }
    }

    /// Apply spectral normalisation and return the normalised weight matrix.
    #[napi]
    pub fn spectral_norm(&mut self) -> Matrix {
        Matrix { inner: facade::gf_op_spectral_norm(&mut self.inner) }
    }

    /// Run self-attention forward pass.
    #[napi]
    pub fn attention(&mut self, inp: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_attention(&inp.inner, &mut self.inner) }
    }

    /// Run self-attention backward pass.
    #[napi]
    pub fn attention_backward(&mut self, grad_out: &Matrix) -> Matrix {
        Matrix { inner: facade::gf_op_attention_backward(&mut self.inner, &grad_out.inner) }
    }

    /// Sanitise weights (replace NaN/Inf with 0).
    #[napi]
    pub fn verify_weights(&mut self) {
        facade::gf_sec_verify_weights(&mut self.inner);
    }
}

// ─── GanMatrixArray ───────────────────────────────────────────────────────────

/// A growable array of Matrix objects, used for FID/IS computation.
#[napi]
pub struct GanMatrixArray {
    inner: facaded_gan_cuda::types::TMatrixArray,
}

#[napi]
impl GanMatrixArray {
    /// Create an empty matrix array.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    /// Append a matrix to the array.
    #[napi]
    pub fn push(&mut self, m: &Matrix) {
        self.inner.push(m.inner.clone());
    }

    /// Return the number of matrices in the array.
    #[napi]
    pub fn len(&self) -> i32 {
        self.inner.len() as i32
    }
}

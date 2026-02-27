/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * PyO3 bindings for facaded_gan_cuda.
 * Exposes two sub-modules:
 *   facaded_gan.facade  — all GF_* façade functions
 *   facaded_gan.api     — direct module functions (matrix, network, loss, …)
 */

use pyo3::prelude::*;
use facaded_gan_cuda::types::*;

// =========================================================================
// Private enum-parsing helpers
// =========================================================================

fn parse_activation(s: &str) -> ActivationType {
    match s {
        "relu"    => ActivationType::ReLU,
        "sigmoid" => ActivationType::Sigmoid,
        "tanh"    => ActivationType::Tanh,
        "leaky"   => ActivationType::LeakyReLU,
        _         => ActivationType::None,
    }
}

fn parse_optimizer(s: &str) -> Optimizer {
    match s {
        "sgd"     => Optimizer::SGD,
        "rmsprop" => Optimizer::RMSProp,
        _         => Optimizer::Adam,
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

fn parse_loss_type(s: &str) -> LossType {
    match s {
        "wgan" | "wgan-gp" => LossType::WGANGP,
        "hinge"            => LossType::Hinge,
        "ls"               => LossType::LeastSquares,
        _                  => LossType::BCE,
    }
}

fn activation_to_str(a: ActivationType) -> &'static str {
    match a {
        ActivationType::ReLU     => "relu",
        ActivationType::Sigmoid  => "sigmoid",
        ActivationType::Tanh     => "tanh",
        ActivationType::LeakyReLU => "leaky",
        ActivationType::None     => "none",
    }
}

fn optimizer_to_str(o: Optimizer) -> &'static str {
    match o {
        Optimizer::Adam    => "adam",
        Optimizer::SGD     => "sgd",
        Optimizer::RMSProp => "rmsprop",
    }
}

fn noise_type_to_str(n: NoiseType) -> &'static str {
    match n {
        NoiseType::Gauss   => "gauss",
        NoiseType::Uniform => "uniform",
        NoiseType::Analog  => "analog",
    }
}

fn data_type_to_str(d: DataType) -> &'static str {
    match d {
        DataType::Image  => "image",
        DataType::Audio  => "audio",
        DataType::Vector => "vector",
    }
}

fn loss_type_to_str(l: LossType) -> &'static str {
    match l {
        LossType::BCE          => "bce",
        LossType::WGANGP       => "wgan",
        LossType::Hinge        => "hinge",
        LossType::LeastSquares => "ls",
    }
}

// =========================================================================
// Python classes
// =========================================================================

/// Python wrapper for GANConfig — all fields exposed as getters/setters.
#[pyclass(name = "GANConfig")]
#[derive(Clone)]
struct PyGANConfig {
    inner: GANConfig,
}

#[pymethods]
impl PyGANConfig {
    #[new]
    fn new() -> Self { PyGANConfig { inner: GANConfig::default() } }

    // --- integer fields ---
    #[getter] fn epochs(&self)         -> i32  { self.inner.epochs }
    #[setter] fn set_epochs(&mut self, v: i32)  { self.inner.epochs = v; }

    #[getter] fn batch_size(&self)     -> i32  { self.inner.batch_size }
    #[setter] fn set_batch_size(&mut self, v: i32) { self.inner.batch_size = v; }

    #[getter] fn generator_bits(&self) -> i32  { self.inner.generator_bits }
    #[setter] fn set_generator_bits(&mut self, v: i32) { self.inner.generator_bits = v; }

    #[getter] fn discriminator_bits(&self) -> i32  { self.inner.discriminator_bits }
    #[setter] fn set_discriminator_bits(&mut self, v: i32) { self.inner.discriminator_bits = v; }

    #[getter] fn noise_depth(&self)    -> i32  { self.inner.noise_depth }
    #[setter] fn set_noise_depth(&mut self, v: i32) { self.inner.noise_depth = v; }

    #[getter] fn condition_size(&self) -> i32  { self.inner.condition_size }
    #[setter] fn set_condition_size(&mut self, v: i32) { self.inner.condition_size = v; }

    #[getter] fn max_res_level(&self)  -> i32  { self.inner.max_res_level }
    #[setter] fn set_max_res_level(&mut self, v: i32) { self.inner.max_res_level = v; }

    #[getter] fn metric_interval(&self) -> i32  { self.inner.metric_interval }
    #[setter] fn set_metric_interval(&mut self, v: i32) { self.inner.metric_interval = v; }

    #[getter] fn checkpoint_interval(&self) -> i32  { self.inner.checkpoint_interval }
    #[setter] fn set_checkpoint_interval(&mut self, v: i32) { self.inner.checkpoint_interval = v; }

    #[getter] fn fuzz_iterations(&self) -> i32  { self.inner.fuzz_iterations }
    #[setter] fn set_fuzz_iterations(&mut self, v: i32) { self.inner.fuzz_iterations = v; }

    #[getter] fn num_threads(&self)    -> i32  { self.inner.num_threads }
    #[setter] fn set_num_threads(&mut self, v: i32) { self.inner.num_threads = v; }

    // --- float fields ---
    #[getter] fn learning_rate(&self)  -> f32  { self.inner.learning_rate }
    #[setter] fn set_learning_rate(&mut self, v: f32) { self.inner.learning_rate = v; }

    #[getter] fn gp_lambda(&self)      -> f32  { self.inner.gp_lambda }
    #[setter] fn set_gp_lambda(&mut self, v: f32) { self.inner.gp_lambda = v; }

    #[getter] fn generator_lr(&self)   -> f32  { self.inner.generator_lr }
    #[setter] fn set_generator_lr(&mut self, v: f32) { self.inner.generator_lr = v; }

    #[getter] fn discriminator_lr(&self) -> f32 { self.inner.discriminator_lr }
    #[setter] fn set_discriminator_lr(&mut self, v: f32) { self.inner.discriminator_lr = v; }

    #[getter] fn weight_decay_val(&self) -> f32  { self.inner.weight_decay_val }
    #[setter] fn set_weight_decay_val(&mut self, v: f32) { self.inner.weight_decay_val = v; }

    // --- bool fields ---
    #[getter] fn use_batch_norm(&self)        -> bool { self.inner.use_batch_norm }
    #[setter] fn set_use_batch_norm(&mut self, v: bool) { self.inner.use_batch_norm = v; }

    #[getter] fn use_layer_norm(&self)        -> bool { self.inner.use_layer_norm }
    #[setter] fn set_use_layer_norm(&mut self, v: bool) { self.inner.use_layer_norm = v; }

    #[getter] fn use_spectral_norm(&self)     -> bool { self.inner.use_spectral_norm }
    #[setter] fn set_use_spectral_norm(&mut self, v: bool) { self.inner.use_spectral_norm = v; }

    #[getter] fn use_label_smoothing(&self)   -> bool { self.inner.use_label_smoothing }
    #[setter] fn set_use_label_smoothing(&mut self, v: bool) { self.inner.use_label_smoothing = v; }

    #[getter] fn use_feature_matching(&self)  -> bool { self.inner.use_feature_matching }
    #[setter] fn set_use_feature_matching(&mut self, v: bool) { self.inner.use_feature_matching = v; }

    #[getter] fn use_minibatch_std_dev(&self) -> bool { self.inner.use_minibatch_std_dev }
    #[setter] fn set_use_minibatch_std_dev(&mut self, v: bool) { self.inner.use_minibatch_std_dev = v; }

    #[getter] fn use_progressive(&self)   -> bool { self.inner.use_progressive }
    #[setter] fn set_use_progressive(&mut self, v: bool) { self.inner.use_progressive = v; }

    #[getter] fn use_augmentation(&self)  -> bool { self.inner.use_augmentation }
    #[setter] fn set_use_augmentation(&mut self, v: bool) { self.inner.use_augmentation = v; }

    #[getter] fn compute_metrics(&self)   -> bool { self.inner.compute_metrics }
    #[setter] fn set_compute_metrics(&mut self, v: bool) { self.inner.compute_metrics = v; }

    #[getter] fn use_weight_decay(&self)  -> bool { self.inner.use_weight_decay }
    #[setter] fn set_use_weight_decay(&mut self, v: bool) { self.inner.use_weight_decay = v; }

    #[getter] fn use_cosine_anneal(&self) -> bool { self.inner.use_cosine_anneal }
    #[setter] fn set_use_cosine_anneal(&mut self, v: bool) { self.inner.use_cosine_anneal = v; }

    #[getter] fn audit_log(&self)         -> bool { self.inner.audit_log }
    #[setter] fn set_audit_log(&mut self, v: bool) { self.inner.audit_log = v; }

    #[getter] fn use_encryption(&self)    -> bool { self.inner.use_encryption }
    #[setter] fn set_use_encryption(&mut self, v: bool) { self.inner.use_encryption = v; }

    #[getter] fn use_conv(&self)          -> bool { self.inner.use_conv }
    #[setter] fn set_use_conv(&mut self, v: bool) { self.inner.use_conv = v; }

    #[getter] fn use_attention(&self)     -> bool { self.inner.use_attention }
    #[setter] fn set_use_attention(&mut self, v: bool) { self.inner.use_attention = v; }

    #[getter] fn run_tests(&self)         -> bool { self.inner.run_tests }
    #[setter] fn set_run_tests(&mut self, v: bool) { self.inner.run_tests = v; }

    #[getter] fn run_fuzz(&self)          -> bool { self.inner.run_fuzz }
    #[setter] fn set_run_fuzz(&mut self, v: bool) { self.inner.run_fuzz = v; }

    #[getter] fn run_quality_tests(&self) -> bool { self.inner.run_quality_tests }
    #[setter] fn set_run_quality_tests(&mut self, v: bool) { self.inner.run_quality_tests = v; }

    // --- string fields ---
    #[getter] fn patch_config(&self)    -> String { self.inner.patch_config.clone() }
    #[setter] fn set_patch_config(&mut self, v: String) { self.inner.patch_config = v; }

    #[getter] fn save_model(&self)      -> String { self.inner.save_model.clone() }
    #[setter] fn set_save_model(&mut self, v: String) { self.inner.save_model = v; }

    #[getter] fn load_model(&self)      -> String { self.inner.load_model.clone() }
    #[setter] fn set_load_model(&mut self, v: String) { self.inner.load_model = v; }

    #[getter] fn load_json_model(&self) -> String { self.inner.load_json_model.clone() }
    #[setter] fn set_load_json_model(&mut self, v: String) { self.inner.load_json_model = v; }

    #[getter] fn output_dir(&self)      -> String { self.inner.output_dir.clone() }
    #[setter] fn set_output_dir(&mut self, v: String) { self.inner.output_dir = v; }

    #[getter] fn data_path(&self)       -> String { self.inner.data_path.clone() }
    #[setter] fn set_data_path(&mut self, v: String) { self.inner.data_path = v; }

    #[getter] fn audit_log_file(&self)  -> String { self.inner.audit_log_file.clone() }
    #[setter] fn set_audit_log_file(&mut self, v: String) { self.inner.audit_log_file = v; }

    #[getter] fn encryption_key(&self)  -> String { self.inner.encryption_key.clone() }
    #[setter] fn set_encryption_key(&mut self, v: String) { self.inner.encryption_key = v; }

    // --- enum fields (string-based) ---
    #[getter] fn activation(&self)  -> &str { activation_to_str(self.inner.activation) }
    #[setter] fn set_activation(&mut self, v: &str) { self.inner.activation = parse_activation(v); }

    #[getter] fn noise_type(&self)  -> &str { noise_type_to_str(self.inner.noise_type) }
    #[setter] fn set_noise_type(&mut self, v: &str) { self.inner.noise_type = parse_noise_type(v); }

    #[getter] fn optimizer(&self)   -> &str { optimizer_to_str(self.inner.optimizer) }
    #[setter] fn set_optimizer(&mut self, v: &str) { self.inner.optimizer = parse_optimizer(v); }

    #[getter] fn loss_type(&self)   -> &str { loss_type_to_str(self.inner.loss_type) }
    #[setter] fn set_loss_type(&mut self, v: &str) { self.inner.loss_type = parse_loss_type(v); }

    #[getter] fn data_type(&self)   -> &str { data_type_to_str(self.inner.data_type) }
    #[setter] fn set_data_type(&mut self, v: &str) { self.inner.data_type = parse_data_type(v); }
}

/// Python wrapper for Network — key read-only fields.
#[pyclass(name = "Network")]
#[derive(Clone)]
struct PyNetwork {
    inner: Network,
}

#[pymethods]
impl PyNetwork {
    #[getter] fn layer_count(&self)    -> i32  { self.inner.layer_count }
    #[getter] fn learning_rate(&self)  -> f32  { self.inner.learning_rate }
    #[getter] fn is_training(&self)    -> bool { self.inner.is_training }
    #[getter] fn optimizer(&self)      -> &str { optimizer_to_str(self.inner.optimizer) }
}

/// Python wrapper for Dataset.
#[pyclass(name = "Dataset")]
#[derive(Clone)]
struct PyDataset {
    inner: Dataset,
}

#[pymethods]
impl PyDataset {
    #[getter] fn count(&self)     -> i32    { self.inner.count }
    #[getter] fn data_type(&self) -> &str   { data_type_to_str(self.inner.data_type) }
}

/// Python wrapper for GANMetrics — all fields as getters.
#[pyclass(name = "GANMetrics")]
#[derive(Clone)]
struct PyGANMetrics {
    inner: GANMetrics,
}

#[pymethods]
impl PyGANMetrics {
    #[getter] fn d_loss_real(&self)  -> f32 { self.inner.d_loss_real }
    #[getter] fn d_loss_fake(&self)  -> f32 { self.inner.d_loss_fake }
    #[getter] fn g_loss(&self)       -> f32 { self.inner.g_loss }
    #[getter] fn fid_score(&self)    -> f32 { self.inner.fid_score }
    #[getter] fn is_score(&self)     -> f32 { self.inner.is_score }
    #[getter] fn grad_penalty(&self) -> f32 { self.inner.grad_penalty }
    #[getter] fn epoch(&self)        -> i32 { self.inner.epoch }
    #[getter] fn batch(&self)        -> i32 { self.inner.batch }

    fn __repr__(&self) -> String {
        format!(
            "GANMetrics(d_loss_real={:.4}, d_loss_fake={:.4}, g_loss={:.4}, epoch={})",
            self.inner.d_loss_real, self.inner.d_loss_fake,
            self.inner.g_loss,      self.inner.epoch,
        )
    }
}

/// Python wrapper for GANResult.
#[pyclass(name = "GANResult")]
struct PyGANResult {
    inner: GANResult,
}

#[pymethods]
impl PyGANResult {
    #[getter]
    fn generator(&self, py: Python<'_>) -> PyResult<Py<PyNetwork>> {
        Py::new(py, PyNetwork { inner: self.inner.generator.clone() })
    }
    #[getter]
    fn discriminator(&self, py: Python<'_>) -> PyResult<Py<PyNetwork>> {
        Py::new(py, PyNetwork { inner: self.inner.discriminator.clone() })
    }
    #[getter]
    fn metrics(&self, py: Python<'_>) -> PyResult<Py<PyGANMetrics>> {
        Py::new(py, PyGANMetrics { inner: self.inner.metrics.clone() })
    }
}

// =========================================================================
// Top-level module functions
// =========================================================================

#[pyfunction]
fn run(py: Python<'_>, config: Bound<'_, PyGANConfig>) -> PyResult<Py<PyGANResult>> {
    let result = facaded_gan_cuda::facade::gf_run(&config.borrow().inner);
    Py::new(py, PyGANResult { inner: result })
}

#[pyfunction]
fn init_backend(name: &str) {
    use facaded_gan_cuda::backend::{self, ComputeBackend};
    let be: ComputeBackend = name.parse().unwrap_or(ComputeBackend::CPU);
    backend::init_backend(be);
}

#[pyfunction]
fn detect_backend() -> String {
    facaded_gan_cuda::backend::detect_best_backend().to_string()
}

// =========================================================================
// facade submodule — GF_* functions
// =========================================================================

// --- GF_Op_ matrix ---
#[pyfunction(name = "gf_op_create_matrix")]
fn fac_gf_op_create_matrix(rows: i32, cols: i32) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_create_matrix(rows, cols)
}
#[pyfunction(name = "gf_op_create_vector")]
fn fac_gf_op_create_vector(size: i32) -> Vec<f32> {
    facaded_gan_cuda::facade::gf_op_create_vector(size)
}
#[pyfunction(name = "gf_op_matrix_multiply")]
fn fac_gf_op_matrix_multiply(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_matrix_multiply(&a, &b)
}
#[pyfunction(name = "gf_op_matrix_add")]
fn fac_gf_op_matrix_add(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_matrix_add(&a, &b)
}
#[pyfunction(name = "gf_op_matrix_subtract")]
fn fac_gf_op_matrix_subtract(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_matrix_subtract(&a, &b)
}
#[pyfunction(name = "gf_op_matrix_scale")]
fn fac_gf_op_matrix_scale(a: Vec<Vec<f32>>, s: f32) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_matrix_scale(&a, s)
}
#[pyfunction(name = "gf_op_matrix_transpose")]
fn fac_gf_op_matrix_transpose(a: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_matrix_transpose(&a)
}
#[pyfunction(name = "gf_op_matrix_normalize")]
fn fac_gf_op_matrix_normalize(a: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_matrix_normalize(&a)
}
#[pyfunction(name = "gf_op_matrix_element_mul")]
fn fac_gf_op_matrix_element_mul(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_matrix_element_mul(&a, &b)
}
#[pyfunction(name = "gf_op_safe_get")]
fn fac_gf_op_safe_get(m: Vec<Vec<f32>>, r: i32, c: i32, def: f32) -> f32 {
    facaded_gan_cuda::facade::gf_op_safe_get(&m, r, c, def)
}

// --- GF_Op_ activations ---
#[pyfunction(name = "gf_op_relu")]
fn fac_gf_op_relu(a: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_relu(&a)
}
#[pyfunction(name = "gf_op_leaky_relu")]
fn fac_gf_op_leaky_relu(a: Vec<Vec<f32>>, alpha: f32) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_leaky_relu(&a, alpha)
}
#[pyfunction(name = "gf_op_sigmoid")]
fn fac_gf_op_sigmoid(a: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_sigmoid(&a)
}
#[pyfunction(name = "gf_op_tanh")]
fn fac_gf_op_tanh(a: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_tanh(&a)
}
#[pyfunction(name = "gf_op_softmax")]
fn fac_gf_op_softmax(a: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_softmax(&a)
}
#[pyfunction(name = "gf_op_activate")]
fn fac_gf_op_activate(a: Vec<Vec<f32>>, act: &str) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_activate(&a, parse_activation(act))
}
#[pyfunction(name = "gf_op_activation_backward")]
fn fac_gf_op_activation_backward(grad_out: Vec<Vec<f32>>, pre_act: Vec<Vec<f32>>, act: &str) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_op_activation_backward(&grad_out, &pre_act, parse_activation(act))
}

// --- GF_Op_ random ---
#[pyfunction(name = "gf_op_random_gaussian")]
fn fac_gf_op_random_gaussian() -> f32 {
    facaded_gan_cuda::facade::gf_op_random_gaussian()
}
#[pyfunction(name = "gf_op_random_uniform")]
fn fac_gf_op_random_uniform(lo: f32, hi: f32) -> f32 {
    facaded_gan_cuda::facade::gf_op_random_uniform(lo, hi)
}
#[pyfunction(name = "gf_op_generate_noise")]
fn fac_gf_op_generate_noise(size: i32, depth: i32, nt: &str) -> Vec<Vec<f32>> {
    let mut noise = vec![];
    facaded_gan_cuda::facade::gf_op_generate_noise(&mut noise, size, depth, parse_noise_type(nt));
    noise
}
#[pyfunction(name = "gf_op_noise_slerp")]
fn fac_gf_op_noise_slerp(v1: Vec<f32>, v2: Vec<f32>, t: f32) -> Vec<f32> {
    facaded_gan_cuda::facade::gf_op_noise_slerp(&v1, &v2, t)
}

// --- GF_Gen_ ---
#[pyfunction(name = "gf_gen_build")]
fn fac_gf_gen_build(py: Python<'_>, sizes: Vec<i32>, act: &str, opt: &str, lr: f32) -> PyResult<Py<PyNetwork>> {
    let net = facaded_gan_cuda::facade::gf_gen_build(&sizes, parse_activation(act), parse_optimizer(opt), lr);
    Py::new(py, PyNetwork { inner: net })
}
#[pyfunction(name = "gf_gen_build_conv")]
fn fac_gf_gen_build_conv(py: Python<'_>, noise_dim: i32, cond_sz: i32, base_ch: i32, act: &str, opt: &str, lr: f32) -> PyResult<Py<PyNetwork>> {
    let net = facaded_gan_cuda::facade::gf_gen_build_conv(noise_dim, cond_sz, base_ch, parse_activation(act), parse_optimizer(opt), lr);
    Py::new(py, PyNetwork { inner: net })
}
#[pyfunction(name = "gf_gen_forward")]
fn fac_gf_gen_forward(gen: Bound<'_, PyNetwork>, inp: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut g = gen.borrow_mut();
    facaded_gan_cuda::facade::gf_gen_forward(&mut g.inner, &inp)
}
#[pyfunction(name = "gf_gen_backward")]
fn fac_gf_gen_backward(gen: Bound<'_, PyNetwork>, grad_out: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut g = gen.borrow_mut();
    facaded_gan_cuda::facade::gf_gen_backward(&mut g.inner, &grad_out)
}
#[pyfunction(name = "gf_gen_sample")]
fn fac_gf_gen_sample(gen: Bound<'_, PyNetwork>, count: i32, noise_dim: i32, nt: &str) -> Vec<Vec<f32>> {
    let mut g = gen.borrow_mut();
    facaded_gan_cuda::facade::gf_gen_sample(&mut g.inner, count, noise_dim, parse_noise_type(nt))
}
#[pyfunction(name = "gf_gen_update_weights")]
fn fac_gf_gen_update_weights(gen: Bound<'_, PyNetwork>) {
    let mut g = gen.borrow_mut();
    facaded_gan_cuda::facade::gf_gen_update_weights(&mut g.inner);
}
#[pyfunction(name = "gf_gen_set_training")]
fn fac_gf_gen_set_training(gen: Bound<'_, PyNetwork>, training: bool) {
    let mut g = gen.borrow_mut();
    facaded_gan_cuda::facade::gf_gen_set_training(&mut g.inner, training);
}
#[pyfunction(name = "gf_gen_noise")]
fn fac_gf_gen_noise(size: i32, depth: i32, nt: &str) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_gen_noise(size, depth, parse_noise_type(nt))
}
#[pyfunction(name = "gf_gen_noise_slerp")]
fn fac_gf_gen_noise_slerp(v1: Vec<f32>, v2: Vec<f32>, t: f32) -> Vec<f32> {
    facaded_gan_cuda::facade::gf_gen_noise_slerp(&v1, &v2, t)
}
#[pyfunction(name = "gf_gen_deep_copy")]
fn fac_gf_gen_deep_copy(py: Python<'_>, gen: Bound<'_, PyNetwork>) -> PyResult<Py<PyNetwork>> {
    let g = gen.borrow();
    let copy = facaded_gan_cuda::facade::gf_gen_deep_copy(&g.inner);
    Py::new(py, PyNetwork { inner: copy })
}
#[pyfunction(name = "gf_gen_get_layer_output")]
fn fac_gf_gen_get_layer_output(gen: Bound<'_, PyNetwork>, idx: i32) -> Vec<Vec<f32>> {
    let g = gen.borrow();
    facaded_gan_cuda::facade::gf_gen_get_layer_output(&g.inner, idx)
}

// --- GF_Disc_ ---
#[pyfunction(name = "gf_disc_build")]
fn fac_gf_disc_build(py: Python<'_>, sizes: Vec<i32>, act: &str, opt: &str, lr: f32) -> PyResult<Py<PyNetwork>> {
    let net = facaded_gan_cuda::facade::gf_disc_build(&sizes, parse_activation(act), parse_optimizer(opt), lr);
    Py::new(py, PyNetwork { inner: net })
}
#[pyfunction(name = "gf_disc_build_conv")]
fn fac_gf_disc_build_conv(py: Python<'_>, in_ch: i32, in_w: i32, in_h: i32, cond_sz: i32, base_ch: i32, act: &str, opt: &str, lr: f32) -> PyResult<Py<PyNetwork>> {
    let net = facaded_gan_cuda::facade::gf_disc_build_conv(in_ch, in_w, in_h, cond_sz, base_ch, parse_activation(act), parse_optimizer(opt), lr);
    Py::new(py, PyNetwork { inner: net })
}
#[pyfunction(name = "gf_disc_evaluate")]
fn fac_gf_disc_evaluate(disc: Bound<'_, PyNetwork>, inp: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut d = disc.borrow_mut();
    facaded_gan_cuda::facade::gf_disc_evaluate(&mut d.inner, &inp)
}
#[pyfunction(name = "gf_disc_forward")]
fn fac_gf_disc_forward(disc: Bound<'_, PyNetwork>, inp: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut d = disc.borrow_mut();
    facaded_gan_cuda::facade::gf_disc_forward(&mut d.inner, &inp)
}
#[pyfunction(name = "gf_disc_backward")]
fn fac_gf_disc_backward(disc: Bound<'_, PyNetwork>, grad_out: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut d = disc.borrow_mut();
    facaded_gan_cuda::facade::gf_disc_backward(&mut d.inner, &grad_out)
}
#[pyfunction(name = "gf_disc_update_weights")]
fn fac_gf_disc_update_weights(disc: Bound<'_, PyNetwork>) {
    let mut d = disc.borrow_mut();
    facaded_gan_cuda::facade::gf_disc_update_weights(&mut d.inner);
}
#[pyfunction(name = "gf_disc_grad_penalty")]
fn fac_gf_disc_grad_penalty(disc: Bound<'_, PyNetwork>, real: Vec<Vec<f32>>, fake: Vec<Vec<f32>>, lambda: f32) -> f32 {
    let mut d = disc.borrow_mut();
    facaded_gan_cuda::facade::gf_disc_grad_penalty(&mut d.inner, &real, &fake, lambda)
}
#[pyfunction(name = "gf_disc_feature_match")]
fn fac_gf_disc_feature_match(disc: Bound<'_, PyNetwork>, real: Vec<Vec<f32>>, fake: Vec<Vec<f32>>, feat_layer: i32) -> f32 {
    let mut d = disc.borrow_mut();
    facaded_gan_cuda::facade::gf_disc_feature_match(&mut d.inner, &real, &fake, feat_layer)
}
#[pyfunction(name = "gf_disc_minibatch_std_dev")]
fn fac_gf_disc_minibatch_std_dev(inp: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_disc_minibatch_std_dev(&inp)
}
#[pyfunction(name = "gf_disc_set_training")]
fn fac_gf_disc_set_training(disc: Bound<'_, PyNetwork>, training: bool) {
    let mut d = disc.borrow_mut();
    facaded_gan_cuda::facade::gf_disc_set_training(&mut d.inner, training);
}
#[pyfunction(name = "gf_disc_deep_copy")]
fn fac_gf_disc_deep_copy(py: Python<'_>, disc: Bound<'_, PyNetwork>) -> PyResult<Py<PyNetwork>> {
    let d = disc.borrow();
    let copy = facaded_gan_cuda::facade::gf_disc_deep_copy(&d.inner);
    Py::new(py, PyNetwork { inner: copy })
}
#[pyfunction(name = "gf_disc_get_layer_output")]
fn fac_gf_disc_get_layer_output(disc: Bound<'_, PyNetwork>, idx: i32) -> Vec<Vec<f32>> {
    let d = disc.borrow();
    facaded_gan_cuda::facade::gf_disc_get_layer_output(&d.inner, idx)
}

// --- GF_Train_ ---
#[pyfunction(name = "gf_train_full")]
fn fac_gf_train_full(py: Python<'_>, gen: Bound<'_, PyNetwork>, disc: Bound<'_, PyNetwork>, ds: Bound<'_, PyDataset>, cfg: Bound<'_, PyGANConfig>) -> PyResult<Py<PyGANMetrics>> {
    let mut g = gen.borrow_mut();
    let mut d = disc.borrow_mut();
    let ds_r = ds.borrow();
    let cfg_r = cfg.borrow();
    let metrics = facaded_gan_cuda::facade::gf_train_full(&mut g.inner, &mut d.inner, &ds_r.inner, &cfg_r.inner);
    Py::new(py, PyGANMetrics { inner: metrics })
}
#[pyfunction(name = "gf_train_step")]
fn fac_gf_train_step(py: Python<'_>, gen: Bound<'_, PyNetwork>, disc: Bound<'_, PyNetwork>, real_batch: Vec<Vec<f32>>, noise: Vec<Vec<f32>>, cfg: Bound<'_, PyGANConfig>) -> PyResult<Py<PyGANMetrics>> {
    let mut g = gen.borrow_mut();
    let mut d = disc.borrow_mut();
    let cfg_r = cfg.borrow();
    let metrics = facaded_gan_cuda::facade::gf_train_step(&mut g.inner, &mut d.inner, &real_batch, &noise, &cfg_r.inner);
    Py::new(py, PyGANMetrics { inner: metrics })
}
#[pyfunction(name = "gf_train_optimize")]
fn fac_gf_train_optimize(net: Bound<'_, PyNetwork>) {
    let mut n = net.borrow_mut();
    facaded_gan_cuda::facade::gf_train_optimize(&mut n.inner);
}
#[pyfunction(name = "gf_train_cosine_anneal")]
fn fac_gf_train_cosine_anneal(epoch: i32, max_ep: i32, base_lr: f32, min_lr: f32) -> f32 {
    facaded_gan_cuda::facade::gf_train_cosine_anneal(epoch, max_ep, base_lr, min_lr)
}
#[pyfunction(name = "gf_train_bce_loss")]
fn fac_gf_train_bce_loss(pred: Vec<Vec<f32>>, target: Vec<Vec<f32>>) -> f32 {
    facaded_gan_cuda::facade::gf_train_bce_loss(&pred, &target)
}
#[pyfunction(name = "gf_train_bce_grad")]
fn fac_gf_train_bce_grad(pred: Vec<Vec<f32>>, target: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_train_bce_grad(&pred, &target)
}
#[pyfunction(name = "gf_train_wgan_disc_loss")]
fn fac_gf_train_wgan_disc_loss(d_real: Vec<Vec<f32>>, d_fake: Vec<Vec<f32>>) -> f32 {
    facaded_gan_cuda::facade::gf_train_wgan_disc_loss(&d_real, &d_fake)
}
#[pyfunction(name = "gf_train_wgan_gen_loss")]
fn fac_gf_train_wgan_gen_loss(d_fake: Vec<Vec<f32>>) -> f32 {
    facaded_gan_cuda::facade::gf_train_wgan_gen_loss(&d_fake)
}
#[pyfunction(name = "gf_train_hinge_disc_loss")]
fn fac_gf_train_hinge_disc_loss(d_real: Vec<Vec<f32>>, d_fake: Vec<Vec<f32>>) -> f32 {
    facaded_gan_cuda::facade::gf_train_hinge_disc_loss(&d_real, &d_fake)
}
#[pyfunction(name = "gf_train_hinge_gen_loss")]
fn fac_gf_train_hinge_gen_loss(d_fake: Vec<Vec<f32>>) -> f32 {
    facaded_gan_cuda::facade::gf_train_hinge_gen_loss(&d_fake)
}
#[pyfunction(name = "gf_train_ls_disc_loss")]
fn fac_gf_train_ls_disc_loss(d_real: Vec<Vec<f32>>, d_fake: Vec<Vec<f32>>) -> f32 {
    facaded_gan_cuda::facade::gf_train_ls_disc_loss(&d_real, &d_fake)
}
#[pyfunction(name = "gf_train_ls_gen_loss")]
fn fac_gf_train_ls_gen_loss(d_fake: Vec<Vec<f32>>) -> f32 {
    facaded_gan_cuda::facade::gf_train_ls_gen_loss(&d_fake)
}
#[pyfunction(name = "gf_train_label_smoothing")]
fn fac_gf_train_label_smoothing(labels: Vec<Vec<f32>>, lo: f32, hi: f32) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_train_label_smoothing(&labels, lo, hi)
}
#[pyfunction(name = "gf_train_load_dataset")]
fn fac_gf_train_load_dataset(py: Python<'_>, path: &str, dt: &str) -> PyResult<Py<PyDataset>> {
    let ds = facaded_gan_cuda::facade::gf_train_load_dataset(path, parse_data_type(dt));
    Py::new(py, PyDataset { inner: ds })
}
#[pyfunction(name = "gf_train_create_synthetic")]
fn fac_gf_train_create_synthetic(py: Python<'_>, count: i32, features: i32) -> PyResult<Py<PyDataset>> {
    let ds = facaded_gan_cuda::facade::gf_train_create_synthetic(count, features);
    Py::new(py, PyDataset { inner: ds })
}
#[pyfunction(name = "gf_train_augment")]
fn fac_gf_train_augment(sample: Vec<Vec<f32>>, dt: &str) -> Vec<Vec<f32>> {
    facaded_gan_cuda::facade::gf_train_augment(&sample, parse_data_type(dt))
}
#[pyfunction(name = "gf_train_compute_fid")]
fn fac_gf_train_compute_fid(real_s: Vec<Vec<Vec<f32>>>, fake_s: Vec<Vec<Vec<f32>>>) -> f32 {
    facaded_gan_cuda::facade::gf_train_compute_fid(&real_s, &fake_s)
}
#[pyfunction(name = "gf_train_compute_is")]
fn fac_gf_train_compute_is(samples: Vec<Vec<Vec<f32>>>) -> f32 {
    facaded_gan_cuda::facade::gf_train_compute_is(&samples)
}
#[pyfunction(name = "gf_train_log_metrics")]
fn fac_gf_train_log_metrics(met: Bound<'_, PyGANMetrics>, filename: &str) {
    let m = met.borrow();
    facaded_gan_cuda::facade::gf_train_log_metrics(&m.inner, filename);
}
#[pyfunction(name = "gf_train_save_model")]
fn fac_gf_train_save_model(net: Bound<'_, PyNetwork>, filename: &str) {
    let n = net.borrow();
    facaded_gan_cuda::facade::gf_train_save_model(&n.inner, filename);
}
#[pyfunction(name = "gf_train_load_model")]
fn fac_gf_train_load_model(net: Bound<'_, PyNetwork>, filename: &str) {
    let mut n = net.borrow_mut();
    facaded_gan_cuda::facade::gf_train_load_model(&mut n.inner, filename);
}
#[pyfunction(name = "gf_train_save_json")]
fn fac_gf_train_save_json(gen: Bound<'_, PyNetwork>, disc: Bound<'_, PyNetwork>, filename: &str) {
    let g = gen.borrow();
    let d = disc.borrow();
    facaded_gan_cuda::facade::gf_train_save_json(&g.inner, &d.inner, filename);
}
#[pyfunction(name = "gf_train_load_json")]
fn fac_gf_train_load_json(gen: Bound<'_, PyNetwork>, disc: Bound<'_, PyNetwork>, filename: &str) {
    let mut g = gen.borrow_mut();
    let mut d = disc.borrow_mut();
    facaded_gan_cuda::facade::gf_train_load_json(&mut g.inner, &mut d.inner, filename);
}
#[pyfunction(name = "gf_train_save_checkpoint")]
fn fac_gf_train_save_checkpoint(gen: Bound<'_, PyNetwork>, disc: Bound<'_, PyNetwork>, ep: i32, dir: &str) {
    let g = gen.borrow();
    let d = disc.borrow();
    facaded_gan_cuda::facade::gf_train_save_checkpoint(&g.inner, &d.inner, ep, dir);
}
#[pyfunction(name = "gf_train_load_checkpoint")]
fn fac_gf_train_load_checkpoint(gen: Bound<'_, PyNetwork>, disc: Bound<'_, PyNetwork>, ep: i32, dir: &str) {
    let mut g = gen.borrow_mut();
    let mut d = disc.borrow_mut();
    facaded_gan_cuda::facade::gf_train_load_checkpoint(&mut g.inner, &mut d.inner, ep, dir);
}
#[pyfunction(name = "gf_train_save_samples")]
fn fac_gf_train_save_samples(gen: Bound<'_, PyNetwork>, ep: i32, dir: &str, noise_dim: i32, nt: &str) {
    let mut g = gen.borrow_mut();
    facaded_gan_cuda::facade::gf_train_save_samples(&mut g.inner, ep, dir, noise_dim, parse_noise_type(nt));
}
#[pyfunction(name = "gf_train_plot_csv")]
fn fac_gf_train_plot_csv(filename: &str, d_loss: Vec<f32>, g_loss: Vec<f32>, cnt: i32) {
    facaded_gan_cuda::facade::gf_train_plot_csv(filename, &d_loss, &g_loss, cnt);
}
#[pyfunction(name = "gf_train_print_bar")]
fn fac_gf_train_print_bar(d_loss: f32, g_loss: f32, width: i32) {
    facaded_gan_cuda::facade::gf_train_print_bar(d_loss, g_loss, width);
}

// --- GF_Sec_ ---
#[pyfunction(name = "gf_sec_audit_log")]
fn fac_gf_sec_audit_log(msg: &str, log_file: &str) {
    facaded_gan_cuda::facade::gf_sec_audit_log(msg, log_file);
}
#[pyfunction(name = "gf_sec_secure_randomize")]
fn fac_gf_sec_secure_randomize() {
    facaded_gan_cuda::facade::gf_sec_secure_randomize();
}
#[pyfunction(name = "gf_sec_get_os_random")]
fn fac_gf_sec_get_os_random() -> u8 {
    facaded_gan_cuda::facade::gf_sec_get_os_random()
}
#[pyfunction(name = "gf_sec_validate_path")]
fn fac_gf_sec_validate_path(path: &str) -> bool {
    facaded_gan_cuda::facade::gf_sec_validate_path(path)
}
#[pyfunction(name = "gf_sec_verify_network")]
fn fac_gf_sec_verify_network(net: Bound<'_, PyNetwork>) {
    let mut n = net.borrow_mut();
    facaded_gan_cuda::facade::gf_sec_verify_network(&mut n.inner);
}
#[pyfunction(name = "gf_sec_encrypt_model")]
fn fac_gf_sec_encrypt_model(in_f: &str, out_f: &str, key: &str) {
    facaded_gan_cuda::facade::gf_sec_encrypt_model(in_f, out_f, key);
}
#[pyfunction(name = "gf_sec_decrypt_model")]
fn fac_gf_sec_decrypt_model(in_f: &str, out_f: &str, key: &str) {
    facaded_gan_cuda::facade::gf_sec_decrypt_model(in_f, out_f, key);
}
#[pyfunction(name = "gf_sec_bounds_check")]
fn fac_gf_sec_bounds_check(m: Vec<Vec<f32>>, r: i32, c: i32) -> bool {
    facaded_gan_cuda::facade::gf_sec_bounds_check(&m, r, c)
}
#[pyfunction(name = "gf_sec_run_tests")]
fn fac_gf_sec_run_tests() -> bool {
    facaded_gan_cuda::facade::gf_sec_run_tests()
}
#[pyfunction(name = "gf_sec_run_fuzz_tests")]
fn fac_gf_sec_run_fuzz_tests(iterations: i32) -> bool {
    facaded_gan_cuda::facade::gf_sec_run_fuzz_tests(iterations)
}

// --- GF_Run_ ---
#[pyfunction(name = "gf_run")]
fn fac_gf_run(py: Python<'_>, config: Bound<'_, PyGANConfig>) -> PyResult<Py<PyGANResult>> {
    let result = facaded_gan_cuda::facade::gf_run(&config.borrow().inner);
    Py::new(py, PyGANResult { inner: result })
}

// =========================================================================
// api submodule — direct module functions
// =========================================================================

// --- Matrix ---
#[pyfunction(name = "create_matrix")]
fn api_create_matrix(rows: i32, cols: i32) -> Vec<Vec<f32>> {
    facaded_gan_cuda::matrix::create_matrix(rows, cols)
}
#[pyfunction(name = "matrix_multiply")]
fn api_matrix_multiply(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::matrix::matrix_multiply(&a, &b)
}
#[pyfunction(name = "matrix_add")]
fn api_matrix_add(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::matrix::matrix_add(&a, &b)
}
#[pyfunction(name = "matrix_subtract")]
fn api_matrix_subtract(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::matrix::matrix_subtract(&a, &b)
}
#[pyfunction(name = "matrix_scale")]
fn api_matrix_scale(a: Vec<Vec<f32>>, s: f32) -> Vec<Vec<f32>> {
    facaded_gan_cuda::matrix::matrix_scale(&a, s)
}
#[pyfunction(name = "matrix_transpose")]
fn api_matrix_transpose(a: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::matrix::matrix_transpose(&a)
}
#[pyfunction(name = "matrix_normalize")]
fn api_matrix_normalize(a: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::matrix::matrix_normalize(&a)
}
#[pyfunction(name = "matrix_element_mul")]
fn api_matrix_element_mul(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::matrix::matrix_element_mul(&a, &b)
}
#[pyfunction(name = "safe_get")]
fn api_safe_get(m: Vec<Vec<f32>>, r: i32, c: i32, def: f32) -> f32 {
    facaded_gan_cuda::matrix::safe_get(&m, r, c, def)
}
#[pyfunction(name = "safe_set")]
fn api_safe_set(mut m: Vec<Vec<f32>>, r: i32, c: i32, val: f32) -> Vec<Vec<f32>> {
    facaded_gan_cuda::matrix::safe_set(&mut m, r, c, val);
    m
}

// --- Activations ---
#[pyfunction(name = "apply_activation")]
fn api_apply_activation(a: Vec<Vec<f32>>, act: &str) -> Vec<Vec<f32>> {
    facaded_gan_cuda::activations::apply_activation(&a, parse_activation(act))
}
#[pyfunction(name = "activation_backward")]
fn api_activation_backward(grad_out: Vec<Vec<f32>>, pre_act: Vec<Vec<f32>>, act: &str) -> Vec<Vec<f32>> {
    facaded_gan_cuda::activations::activation_backward(&grad_out, &pre_act, parse_activation(act))
}
#[pyfunction(name = "matrix_relu")]
fn api_matrix_relu(a: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::activations::matrix_relu(&a)
}
#[pyfunction(name = "matrix_leaky_relu")]
fn api_matrix_leaky_relu(a: Vec<Vec<f32>>, alpha: f32) -> Vec<Vec<f32>> {
    facaded_gan_cuda::activations::matrix_leaky_relu(&a, alpha)
}
#[pyfunction(name = "matrix_sigmoid")]
fn api_matrix_sigmoid(a: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::activations::matrix_sigmoid(&a)
}
#[pyfunction(name = "matrix_tanh")]
fn api_matrix_tanh(a: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::activations::matrix_tanh(&a)
}
#[pyfunction(name = "matrix_softmax")]
fn api_matrix_softmax(a: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::activations::matrix_softmax(&a)
}

// --- Network ---
#[pyfunction(name = "create_network")]
fn api_create_network(py: Python<'_>, sizes: Vec<i32>, act: &str, opt: &str, lr: f32) -> PyResult<Py<PyNetwork>> {
    let net = facaded_gan_cuda::network::create_network(&sizes, parse_activation(act), parse_optimizer(opt), lr);
    Py::new(py, PyNetwork { inner: net })
}
#[pyfunction(name = "network_forward")]
fn api_network_forward(net: Bound<'_, PyNetwork>, inp: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut n = net.borrow_mut();
    facaded_gan_cuda::network::network_forward(&mut n.inner, &inp)
}
#[pyfunction(name = "network_backward")]
fn api_network_backward(net: Bound<'_, PyNetwork>, grad_out: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut n = net.borrow_mut();
    facaded_gan_cuda::network::network_backward(&mut n.inner, &grad_out)
}
#[pyfunction(name = "network_update_weights")]
fn api_network_update_weights(net: Bound<'_, PyNetwork>) {
    let mut n = net.borrow_mut();
    facaded_gan_cuda::network::network_update_weights(&mut n.inner);
}
#[pyfunction(name = "set_network_training")]
fn api_set_network_training(net: Bound<'_, PyNetwork>, training: bool) {
    let mut n = net.borrow_mut();
    facaded_gan_cuda::network::set_network_training(&mut n.inner, training);
}

// --- Loss ---
#[pyfunction(name = "binary_cross_entropy")]
fn api_binary_cross_entropy(pred: Vec<Vec<f32>>, target: Vec<Vec<f32>>) -> f32 {
    facaded_gan_cuda::loss::binary_cross_entropy(&pred, &target)
}
#[pyfunction(name = "bce_gradient")]
fn api_bce_gradient(pred: Vec<Vec<f32>>, target: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    facaded_gan_cuda::loss::bce_gradient(&pred, &target)
}
#[pyfunction(name = "wgan_disc_loss")]
fn api_wgan_disc_loss(d_real: Vec<Vec<f32>>, d_fake: Vec<Vec<f32>>) -> f32 {
    facaded_gan_cuda::loss::wgan_disc_loss(&d_real, &d_fake)
}
#[pyfunction(name = "wgan_gen_loss")]
fn api_wgan_gen_loss(d_fake: Vec<Vec<f32>>) -> f32 {
    facaded_gan_cuda::loss::wgan_gen_loss(&d_fake)
}
#[pyfunction(name = "hinge_disc_loss")]
fn api_hinge_disc_loss(d_real: Vec<Vec<f32>>, d_fake: Vec<Vec<f32>>) -> f32 {
    facaded_gan_cuda::loss::hinge_disc_loss(&d_real, &d_fake)
}
#[pyfunction(name = "hinge_gen_loss")]
fn api_hinge_gen_loss(d_fake: Vec<Vec<f32>>) -> f32 {
    facaded_gan_cuda::loss::hinge_gen_loss(&d_fake)
}
#[pyfunction(name = "ls_disc_loss")]
fn api_ls_disc_loss(d_real: Vec<Vec<f32>>, d_fake: Vec<Vec<f32>>) -> f32 {
    facaded_gan_cuda::loss::ls_disc_loss(&d_real, &d_fake)
}
#[pyfunction(name = "ls_gen_loss")]
fn api_ls_gen_loss(d_fake: Vec<Vec<f32>>) -> f32 {
    facaded_gan_cuda::loss::ls_gen_loss(&d_fake)
}

// --- Optimizer ---
#[pyfunction(name = "cosine_anneal")]
fn api_cosine_anneal(epoch: i32, max_ep: i32, base_lr: f32, min_lr: f32) -> f32 {
    facaded_gan_cuda::optimizer::cosine_anneal(epoch, max_ep, base_lr, min_lr)
}

// --- Random ---
#[pyfunction(name = "random_gaussian")]
fn api_random_gaussian() -> f32 {
    facaded_gan_cuda::random::random_gaussian()
}
#[pyfunction(name = "random_uniform")]
fn api_random_uniform(lo: f32, hi: f32) -> f32 {
    facaded_gan_cuda::random::random_uniform(lo, hi)
}
#[pyfunction(name = "generate_noise")]
fn api_generate_noise(size: i32, depth: i32, nt: &str) -> Vec<Vec<f32>> {
    let mut noise = vec![];
    facaded_gan_cuda::random::generate_noise(&mut noise, size, depth, parse_noise_type(nt));
    noise
}
#[pyfunction(name = "noise_slerp")]
fn api_noise_slerp(v1: Vec<f32>, v2: Vec<f32>, t: f32) -> Vec<f32> {
    facaded_gan_cuda::random::noise_slerp(&v1, &v2, t)
}

// --- Security ---
#[pyfunction(name = "audit_log")]
fn api_audit_log(msg: &str, log_file: &str) {
    facaded_gan_cuda::security::audit_log(msg, log_file);
}
#[pyfunction(name = "validate_path")]
fn api_validate_path(path: &str) -> bool {
    facaded_gan_cuda::security::validate_path(path)
}
#[pyfunction(name = "verify_network")]
fn api_verify_network(net: Bound<'_, PyNetwork>) {
    let mut n = net.borrow_mut();
    facaded_gan_cuda::security::verify_network(&mut n.inner);
}
#[pyfunction(name = "bounds_check")]
fn api_bounds_check(m: Vec<Vec<f32>>, r: i32, c: i32) -> bool {
    facaded_gan_cuda::security::bounds_check(&m, r, c)
}

// =========================================================================
// Submodule registration helpers
// =========================================================================

fn add_facade_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // GF_Op_
    m.add_function(wrap_pyfunction!(fac_gf_op_create_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_create_vector, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_matrix_multiply, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_matrix_add, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_matrix_subtract, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_matrix_scale, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_matrix_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_matrix_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_matrix_element_mul, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_safe_get, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_relu, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_leaky_relu, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_tanh, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_activate, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_activation_backward, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_random_gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_random_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_generate_noise, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_op_noise_slerp, m)?)?;
    // GF_Gen_
    m.add_function(wrap_pyfunction!(fac_gf_gen_build, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_gen_build_conv, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_gen_forward, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_gen_backward, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_gen_sample, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_gen_update_weights, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_gen_set_training, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_gen_noise, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_gen_noise_slerp, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_gen_deep_copy, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_gen_get_layer_output, m)?)?;
    // GF_Disc_
    m.add_function(wrap_pyfunction!(fac_gf_disc_build, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_disc_build_conv, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_disc_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_disc_forward, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_disc_backward, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_disc_update_weights, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_disc_grad_penalty, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_disc_feature_match, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_disc_minibatch_std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_disc_set_training, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_disc_deep_copy, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_disc_get_layer_output, m)?)?;
    // GF_Train_
    m.add_function(wrap_pyfunction!(fac_gf_train_full, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_step, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_cosine_anneal, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_bce_loss, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_bce_grad, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_wgan_disc_loss, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_wgan_gen_loss, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_hinge_disc_loss, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_hinge_gen_loss, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_ls_disc_loss, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_ls_gen_loss, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_label_smoothing, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_load_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_create_synthetic, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_augment, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_compute_fid, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_compute_is, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_log_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_save_model, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_load_model, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_save_json, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_load_json, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_save_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_load_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_save_samples, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_plot_csv, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_train_print_bar, m)?)?;
    // GF_Sec_
    m.add_function(wrap_pyfunction!(fac_gf_sec_audit_log, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_sec_secure_randomize, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_sec_get_os_random, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_sec_validate_path, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_sec_verify_network, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_sec_encrypt_model, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_sec_decrypt_model, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_sec_bounds_check, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_sec_run_tests, m)?)?;
    m.add_function(wrap_pyfunction!(fac_gf_sec_run_fuzz_tests, m)?)?;
    // GF_Run_
    m.add_function(wrap_pyfunction!(fac_gf_run, m)?)?;
    Ok(())
}

fn add_api_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Matrix
    m.add_function(wrap_pyfunction!(api_create_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(api_matrix_multiply, m)?)?;
    m.add_function(wrap_pyfunction!(api_matrix_add, m)?)?;
    m.add_function(wrap_pyfunction!(api_matrix_subtract, m)?)?;
    m.add_function(wrap_pyfunction!(api_matrix_scale, m)?)?;
    m.add_function(wrap_pyfunction!(api_matrix_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(api_matrix_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(api_matrix_element_mul, m)?)?;
    m.add_function(wrap_pyfunction!(api_safe_get, m)?)?;
    m.add_function(wrap_pyfunction!(api_safe_set, m)?)?;
    // Activations
    m.add_function(wrap_pyfunction!(api_apply_activation, m)?)?;
    m.add_function(wrap_pyfunction!(api_activation_backward, m)?)?;
    m.add_function(wrap_pyfunction!(api_matrix_relu, m)?)?;
    m.add_function(wrap_pyfunction!(api_matrix_leaky_relu, m)?)?;
    m.add_function(wrap_pyfunction!(api_matrix_sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(api_matrix_tanh, m)?)?;
    m.add_function(wrap_pyfunction!(api_matrix_softmax, m)?)?;
    // Network
    m.add_function(wrap_pyfunction!(api_create_network, m)?)?;
    m.add_function(wrap_pyfunction!(api_network_forward, m)?)?;
    m.add_function(wrap_pyfunction!(api_network_backward, m)?)?;
    m.add_function(wrap_pyfunction!(api_network_update_weights, m)?)?;
    m.add_function(wrap_pyfunction!(api_set_network_training, m)?)?;
    // Loss
    m.add_function(wrap_pyfunction!(api_binary_cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(api_bce_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(api_wgan_disc_loss, m)?)?;
    m.add_function(wrap_pyfunction!(api_wgan_gen_loss, m)?)?;
    m.add_function(wrap_pyfunction!(api_hinge_disc_loss, m)?)?;
    m.add_function(wrap_pyfunction!(api_hinge_gen_loss, m)?)?;
    m.add_function(wrap_pyfunction!(api_ls_disc_loss, m)?)?;
    m.add_function(wrap_pyfunction!(api_ls_gen_loss, m)?)?;
    // Optimizer
    m.add_function(wrap_pyfunction!(api_cosine_anneal, m)?)?;
    // Random
    m.add_function(wrap_pyfunction!(api_random_gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(api_random_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(api_generate_noise, m)?)?;
    m.add_function(wrap_pyfunction!(api_noise_slerp, m)?)?;
    // Security
    m.add_function(wrap_pyfunction!(api_audit_log, m)?)?;
    m.add_function(wrap_pyfunction!(api_validate_path, m)?)?;
    m.add_function(wrap_pyfunction!(api_verify_network, m)?)?;
    m.add_function(wrap_pyfunction!(api_bounds_check, m)?)?;
    Ok(())
}

// =========================================================================
// Root module
// =========================================================================

#[pymodule]
fn facaded_gan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    // Python classes
    m.add_class::<PyGANConfig>()?;
    m.add_class::<PyNetwork>()?;
    m.add_class::<PyDataset>()?;
    m.add_class::<PyGANMetrics>()?;
    m.add_class::<PyGANResult>()?;

    // Top-level functions
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(init_backend, m)?)?;
    m.add_function(wrap_pyfunction!(detect_backend, m)?)?;

    // `facade` submodule
    let facade_mod = PyModule::new_bound(py, "facade")?;
    add_facade_submodule(&facade_mod)?;
    m.add_submodule(&facade_mod)?;
    // Register in sys.modules so `from facaded_gan import facade` works
    py.import_bound("sys")?
        .getattr("modules")?
        .set_item("facaded_gan.facade", &facade_mod)?;

    // `api` submodule
    let api_mod = PyModule::new_bound(py, "api")?;
    add_api_submodule(&api_mod)?;
    m.add_submodule(&api_mod)?;
    py.import_bound("sys")?
        .getattr("modules")?
        .set_item("facaded_gan.api", &api_mod)?;

    Ok(())
}

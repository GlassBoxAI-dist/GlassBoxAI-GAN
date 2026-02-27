/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Type definitions matching gan.h
 */

use serde::{Deserialize, Serialize};

pub const GAN_VERSION: &str = "2.0";
pub const DEFAULT_GP_LAMBDA: f32 = 10.0;
pub const DEFAULT_BN_EPS: f32 = 1e-5;
pub const DEFAULT_BN_MOM: f32 = 0.1;
pub const MAX_SPECTRAL_ITER: i32 = 1;
pub const DEFAULT_AUDIT_LOG: &str = "gan_audit.log";

pub type TVector = Vec<f32>;
pub type TMatrix = Vec<Vec<f32>>;
pub type TMatrixArray = Vec<TMatrix>;
pub type TKernelArray = Vec<TMatrix>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum LayerType {
    Dense,
    Conv2D,
    Deconv2D,
    Conv1D,
    BatchNorm,
    LayerNorm,
    SpectralNorm,
    Attention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum LossType {
    BCE,
    WGANGP,
    Hinge,
    LeastSquares,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum DataType {
    Image,
    Audio,
    Vector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum NoiseType {
    Gauss,
    Uniform,
    Analog,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum Optimizer {
    Adam,
    SGD,
    RMSProp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub layer_type: LayerType,
    pub activation: ActivationType,
    pub input_size: i32,
    pub output_size: i32,
    // Dense
    pub weights: TMatrix,
    pub bias: TVector,
    // Conv
    pub kernels: TKernelArray,
    pub kernel_size: i32,
    pub stride: i32,
    pub padding: i32,
    pub in_channels: i32,
    pub out_channels: i32,
    pub in_width: i32,
    pub in_height: i32,
    pub out_width: i32,
    pub out_height: i32,
    // BatchNorm / LayerNorm
    pub bn_gamma: TVector,
    pub bn_beta: TVector,
    pub running_mean: TVector,
    pub running_var: TVector,
    pub bn_epsilon: f32,
    pub bn_momentum: f32,
    // Attention
    pub wq: TMatrix,
    pub wk: TMatrix,
    pub wv: TMatrix,
    pub wo: TMatrix,
    pub num_heads: i32,
    pub head_dim: i32,
    // Spectral norm
    pub spectral_u: TVector,
    pub spectral_v: TVector,
    pub spectral_sigma: f32,
    // Forward cache
    pub layer_input: TMatrix,
    pub layer_output: TMatrix,
    pub pre_activation: TMatrix,
    pub is_training: bool,
    pub cached_q: TMatrix,
    pub cached_k: TMatrix,
    pub cached_v: TMatrix,
    pub cached_scores: TMatrix,
    pub cached_attended: TMatrix,
    pub cached_normalized: TMatrix,
    pub cached_mean: TVector,
    pub cached_var: TVector,
    pub use_checkpoint: bool,
    // Gradients
    pub weight_grad: TMatrix,
    pub bias_grad: TVector,
    pub kernel_grad: TKernelArray,
    pub bn_gamma_grad: TVector,
    pub bn_beta_grad: TVector,
    pub wq_grad: TMatrix,
    pub wk_grad: TMatrix,
    pub wv_grad: TMatrix,
    pub wo_grad: TMatrix,
    // Adam moments
    pub adam_t: i32,
    pub m_weight: TMatrix,
    pub v_weight: TMatrix,
    pub m_bias: TVector,
    pub v_bias: TVector,
    pub m_kernel: TKernelArray,
    pub v_kernel: TKernelArray,
    pub m_bn_gamma: TVector,
    pub v_bn_gamma: TVector,
    pub m_bn_beta: TVector,
    pub v_bn_beta: TVector,
    pub m_wq: TMatrix,
    pub v_wq: TMatrix,
    pub m_wk: TMatrix,
    pub v_wk: TMatrix,
    pub m_wv: TMatrix,
    pub v_wv: TMatrix,
    pub m_wo: TMatrix,
    pub v_wo: TMatrix,
    // RMSProp
    pub rms_weight: TMatrix,
    pub rms_bias: TVector,
}

impl Default for Layer {
    fn default() -> Self {
        Self {
            layer_type: LayerType::Dense,
            activation: ActivationType::None,
            input_size: 0,
            output_size: 0,
            weights: vec![],
            bias: vec![],
            kernels: vec![],
            kernel_size: 0,
            stride: 0,
            padding: 0,
            in_channels: 0,
            out_channels: 0,
            in_width: 0,
            in_height: 0,
            out_width: 0,
            out_height: 0,
            bn_gamma: vec![],
            bn_beta: vec![],
            running_mean: vec![],
            running_var: vec![],
            bn_epsilon: 0.0,
            bn_momentum: 0.0,
            wq: vec![],
            wk: vec![],
            wv: vec![],
            wo: vec![],
            num_heads: 0,
            head_dim: 0,
            spectral_u: vec![],
            spectral_v: vec![],
            spectral_sigma: 0.0,
            layer_input: vec![],
            layer_output: vec![],
            pre_activation: vec![],
            is_training: true,
            cached_q: vec![],
            cached_k: vec![],
            cached_v: vec![],
            cached_scores: vec![],
            cached_attended: vec![],
            cached_normalized: vec![],
            cached_mean: vec![],
            cached_var: vec![],
            use_checkpoint: false,
            weight_grad: vec![],
            bias_grad: vec![],
            kernel_grad: vec![],
            bn_gamma_grad: vec![],
            bn_beta_grad: vec![],
            wq_grad: vec![],
            wk_grad: vec![],
            wv_grad: vec![],
            wo_grad: vec![],
            adam_t: 0,
            m_weight: vec![],
            v_weight: vec![],
            m_bias: vec![],
            v_bias: vec![],
            m_kernel: vec![],
            v_kernel: vec![],
            m_bn_gamma: vec![],
            v_bn_gamma: vec![],
            m_bn_beta: vec![],
            v_bn_beta: vec![],
            m_wq: vec![],
            v_wq: vec![],
            m_wk: vec![],
            v_wk: vec![],
            m_wv: vec![],
            v_wv: vec![],
            m_wo: vec![],
            v_wo: vec![],
            rms_weight: vec![],
            rms_bias: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub layer_count: i32,
    pub optimizer: Optimizer,
    pub learning_rate: f32,
    pub momentum: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub progressive_alpha: f32,
    pub current_res_level: i32,
    pub is_training: bool,
}

impl Default for Network {
    fn default() -> Self {
        Self {
            layers: vec![],
            layer_count: 0,
            optimizer: Optimizer::Adam,
            learning_rate: 0.0,
            momentum: 0.0,
            beta1: 0.0,
            beta2: 0.0,
            epsilon: 0.0,
            weight_decay: 0.0,
            progressive_alpha: 0.0,
            current_res_level: 0,
            is_training: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GANConfig {
    pub epochs: i32,
    pub batch_size: i32,
    pub generator_bits: i32,
    pub discriminator_bits: i32,
    pub activation: ActivationType,
    pub noise_type: NoiseType,
    pub noise_depth: i32,
    pub patch_config: String,
    pub save_model: String,
    pub load_model: String,
    pub load_json_model: String,
    pub output_dir: String,
    pub learning_rate: f32,
    pub optimizer: Optimizer,
    pub loss_type: LossType,
    pub gp_lambda: f32,
    pub condition_size: i32,
    pub use_batch_norm: bool,
    pub use_layer_norm: bool,
    pub use_spectral_norm: bool,
    pub use_label_smoothing: bool,
    pub use_feature_matching: bool,
    pub use_minibatch_std_dev: bool,
    pub generator_lr: f32,
    pub discriminator_lr: f32,
    pub use_progressive: bool,
    pub max_res_level: i32,
    pub data_type: DataType,
    pub data_path: String,
    pub use_augmentation: bool,
    pub compute_metrics: bool,
    pub metric_interval: i32,
    pub use_weight_decay: bool,
    pub weight_decay_val: f32,
    pub use_cosine_anneal: bool,
    pub audit_log: bool,
    pub audit_log_file: String,
    pub checkpoint_interval: i32,
    pub use_encryption: bool,
    pub encryption_key: String,
    pub use_conv: bool,
    pub use_attention: bool,
    pub run_tests: bool,
    pub run_fuzz: bool,
    pub fuzz_iterations: i32,
    pub run_quality_tests: bool,
    pub num_threads: i32,
}

impl Default for GANConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            generator_bits: 0,
            discriminator_bits: 0,
            activation: ActivationType::LeakyReLU,
            noise_type: NoiseType::Gauss,
            noise_depth: 64,
            patch_config: String::new(),
            save_model: String::new(),
            load_model: String::new(),
            load_json_model: String::new(),
            output_dir: String::from("gan_output"),
            learning_rate: 0.0002,
            optimizer: Optimizer::Adam,
            loss_type: LossType::BCE,
            gp_lambda: DEFAULT_GP_LAMBDA,
            condition_size: 0,
            use_batch_norm: false,
            use_layer_norm: false,
            use_spectral_norm: false,
            use_label_smoothing: false,
            use_feature_matching: false,
            use_minibatch_std_dev: false,
            generator_lr: 0.0,
            discriminator_lr: 0.0,
            use_progressive: false,
            max_res_level: 0,
            data_type: DataType::Vector,
            data_path: String::new(),
            use_augmentation: false,
            compute_metrics: false,
            metric_interval: 0,
            use_weight_decay: false,
            weight_decay_val: 0.0,
            use_cosine_anneal: false,
            audit_log: false,
            audit_log_file: String::from(DEFAULT_AUDIT_LOG),
            checkpoint_interval: 0,
            use_encryption: false,
            encryption_key: String::new(),
            use_conv: false,
            use_attention: false,
            run_tests: false,
            run_fuzz: false,
            fuzz_iterations: 0,
            run_quality_tests: false,
            num_threads: 0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GANMetrics {
    pub d_loss_real: f32,
    pub d_loss_fake: f32,
    pub g_loss: f32,
    pub fid_score: f32,
    pub is_score: f32,
    pub grad_penalty: f32,
    pub epoch: i32,
    pub batch: i32,
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct GANResult {
    pub generator:     Network,
    pub discriminator: Network,
    pub metrics:       GANMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub samples: TMatrixArray,
    pub labels: TMatrix,
    pub count: i32,
    pub data_type: DataType,
    pub sample_width: i32,
    pub sample_height: i32,
    pub sample_channels: i32,
}

impl Default for Dataset {
    fn default() -> Self {
        Self {
            samples: vec![],
            labels: vec![],
            count: 0,
            data_type: DataType::Vector,
            sample_width: 0,
            sample_height: 0,
            sample_channels: 0,
        }
    }
}

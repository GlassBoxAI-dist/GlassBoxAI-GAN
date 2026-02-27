/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Backend abstraction — CPU, CUDA, OpenCL, Hybrid compute backends.
 */

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "opencl")]
pub mod opencl;

use crate::types::TMatrix;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    CPU,
    CUDA,
    OpenCL,
    Hybrid,
}

impl std::fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeBackend::CPU => write!(f, "CPU"),
            ComputeBackend::CUDA => write!(f, "CUDA"),
            ComputeBackend::OpenCL => write!(f, "OpenCL"),
            ComputeBackend::Hybrid => write!(f, "Hybrid"),
        }
    }
}

impl std::str::FromStr for ComputeBackend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(ComputeBackend::CPU),
            "cuda" => Ok(ComputeBackend::CUDA),
            "opencl" | "cl" => Ok(ComputeBackend::OpenCL),
            "hybrid" | "auto" => Ok(ComputeBackend::Hybrid),
            _ => Err(format!("Unknown backend: {}", s)),
        }
    }
}

/// Trait for GPU-acceleratable operations.
/// Flat f32 slices are used; callers flatten/unflatten TMatrix as needed.
pub trait BackendOps: Send + Sync {
    fn name(&self) -> &str;

    // Matrix ops
    fn matrix_multiply(&self, a: &[f32], b: &[f32], m: i32, k: i32, n: i32) -> Vec<f32>;
    fn matrix_add(&self, a: &[f32], b: &[f32], len: i32) -> Vec<f32>;
    fn matrix_sub(&self, a: &[f32], b: &[f32], len: i32) -> Vec<f32>;
    fn matrix_scale(&self, a: &[f32], s: f32, len: i32) -> Vec<f32>;
    fn matrix_element_mul(&self, a: &[f32], b: &[f32], len: i32) -> Vec<f32>;
    fn matrix_add_inplace(&self, a: &mut [f32], b: &[f32], len: i32);
    fn matrix_scale_inplace(&self, a: &mut [f32], s: f32, len: i32);
    fn matrix_clip_inplace(&self, a: &mut [f32], lo: f32, hi: f32, len: i32);

    // Activations
    fn relu_forward(&self, a: &[f32], len: i32) -> Vec<f32>;
    fn leaky_relu_forward(&self, a: &[f32], alpha: f32, len: i32) -> Vec<f32>;
    fn sigmoid_forward(&self, a: &[f32], len: i32) -> Vec<f32>;
    fn tanh_forward(&self, a: &[f32], len: i32) -> Vec<f32>;
    fn activation_backward(&self, grad: &[f32], pre_act: &[f32], act_type: i32, len: i32)
        -> Vec<f32>;

    // Optimizers
    fn adam_update(
        &self, p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32],
        t: i32, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, len: i32,
    );
    fn sgd_update(&self, p: &mut [f32], g: &[f32], lr: f32, wd: f32, len: i32);
    fn rmsprop_update(
        &self, p: &mut [f32], g: &[f32], cache: &mut [f32],
        lr: f32, decay: f32, eps: f32, wd: f32, len: i32,
    );

    // Loss
    fn bce_gradient(&self, pred: &[f32], target: &[f32], len: i32) -> Vec<f32>;
}

// =========================================================================
// Helpers: flatten / unflatten TMatrix ↔ Vec<f32>
// =========================================================================

pub fn flatten_matrix(m: &TMatrix) -> Vec<f32> {
    if m.is_empty() { return vec![]; }
    let rows = m.len();
    let cols = m[0].len();
    let mut flat = Vec::with_capacity(rows * cols);
    for row in m {
        flat.extend_from_slice(row);
    }
    flat
}

pub fn unflatten_matrix(flat: &[f32], rows: usize, cols: usize) -> TMatrix {
    let mut m = Vec::with_capacity(rows);
    for i in 0..rows {
        let start = i * cols;
        let end = start + cols;
        if end <= flat.len() {
            m.push(flat[start..end].to_vec());
        } else {
            m.push(vec![0.0; cols]);
        }
    }
    m
}

// =========================================================================
// Auto-detection & factory
// =========================================================================

/// Detect the best available backend. Tries CUDA first, then OpenCL, then CPU.
pub fn detect_best_backend() -> ComputeBackend {
    #[cfg(feature = "cuda")]
    {
        if cuda_available() {
            return ComputeBackend::CUDA;
        }
    }
    #[cfg(feature = "opencl")]
    {
        if opencl_available() {
            return ComputeBackend::OpenCL;
        }
    }
    ComputeBackend::CPU
}

#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    cudarc::driver::CudaDevice::new(0).is_ok()
}

#[cfg(feature = "opencl")]
fn opencl_available() -> bool {
    !ocl::Platform::list().is_empty()
}

/// Create a backend instance for the given type.
pub fn create_backend(backend: ComputeBackend) -> Box<dyn BackendOps> {
    match backend {
        ComputeBackend::CPU => Box::new(cpu::CpuBackend),
        #[cfg(feature = "cuda")]
        ComputeBackend::CUDA => match cuda::CudaBackend::new() {
            Ok(b) => Box::new(b),
            Err(e) => {
                eprintln!("[CUDA] Init failed ({}), falling back to CPU", e);
                Box::new(cpu::CpuBackend)
            }
        },
        #[cfg(not(feature = "cuda"))]
        ComputeBackend::CUDA => {
            eprintln!("[CUDA] Not compiled in, falling back to CPU");
            Box::new(cpu::CpuBackend)
        }
        #[cfg(feature = "opencl")]
        ComputeBackend::OpenCL => match opencl::OpenCLBackend::new() {
            Ok(b) => Box::new(b),
            Err(e) => {
                eprintln!("[OpenCL] Init failed ({}), falling back to CPU", e);
                Box::new(cpu::CpuBackend)
            }
        },
        #[cfg(not(feature = "opencl"))]
        ComputeBackend::OpenCL => {
            eprintln!("[OpenCL] Not compiled in, falling back to CPU");
            Box::new(cpu::CpuBackend)
        }
        ComputeBackend::Hybrid => {
            // Pick best available for hybrid
            let best = detect_best_backend();
            if best == ComputeBackend::CPU {
                eprintln!("[Hybrid] No GPU found, using CPU");
            } else {
                eprintln!("[Hybrid] Using {} as primary", best);
            }
            create_backend(best)
        }
    }
}

// =========================================================================
// Global backend state
// =========================================================================

use std::sync::OnceLock;

static GLOBAL_BACKEND: OnceLock<Box<dyn BackendOps>> = OnceLock::new();

pub fn init_backend(backend: ComputeBackend) {
    let _ = GLOBAL_BACKEND.set(create_backend(backend));
}

pub fn get_backend() -> &'static dyn BackendOps {
    GLOBAL_BACKEND.get_or_init(|| {
        let best = detect_best_backend();
        eprintln!("[Backend] Auto-detected: {}", best);
        create_backend(best)
    }).as_ref()
}

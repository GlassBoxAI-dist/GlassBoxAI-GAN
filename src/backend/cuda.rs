/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * CUDA backend via cudarc — GPU kernels ported from OpenCL.
 */

use super::BackendOps;
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

const CUDA_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void matrix_multiply(const float* A, const float* B,
                                           float* C, int M, int K, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float s = 0.0f;
        for (int k = 0; k < K; k++) s += A[row * K + k] * B[k * N + col];
        C[row * N + col] = s;
    }
}

extern "C" __global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

extern "C" __global__ void matrix_sub(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] - B[i];
}

extern "C" __global__ void matrix_scale_k(const float* A, float* C, float s, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] * s;
}

extern "C" __global__ void matrix_element_mul(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] * B[i];
}

extern "C" __global__ void matrix_add_inplace(float* A, const float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) A[i] += B[i];
}

extern "C" __global__ void matrix_scale_inplace(float* A, float s, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) A[i] *= s;
}

extern "C" __global__ void matrix_clip_inplace(float* A, float lo, float hi, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (A[i] < lo) A[i] = lo;
        if (A[i] > hi) A[i] = hi;
    }
}

extern "C" __global__ void relu_forward(const float* A, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] > 0.0f ? A[i] : 0.0f;
}

extern "C" __global__ void leaky_relu_forward(const float* A, float* C, float alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] > 0.0f ? A[i] : alpha * A[i];
}

extern "C" __global__ void sigmoid_forward(const float* A, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = A[i];
        if (v > 20.0f) C[i] = 1.0f;
        else if (v < -20.0f) C[i] = 0.0f;
        else C[i] = 1.0f / (1.0f + expf(-v));
    }
}

extern "C" __global__ void tanh_forward(const float* A, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = tanhf(A[i]);
}

// actType: 0=ReLU, 1=LeakyReLU, 2=Sigmoid, 3=Tanh, 4=None
extern "C" __global__ void activation_backward(const float* gradOut, const float* preAct,
                                                float* gradIn, int actType, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float g = gradOut[i], p = preAct[i];
        if (actType == 0)
            gradIn[i] = p > 0.0f ? g : 0.0f;
        else if (actType == 1)
            gradIn[i] = p > 0.0f ? g : 0.01f * g;
        else if (actType == 2) {
            float s = 1.0f / (1.0f + expf(-p));
            gradIn[i] = g * s * (1.0f - s);
        } else if (actType == 3) {
            float s = tanhf(p);
            gradIn[i] = g * (1.0f - s * s);
        } else
            gradIn[i] = g;
    }
}

extern "C" __global__ void adam_update(float* p, const float* g,
                                       float* m, float* v,
                                       int t, float lr, float b1, float b2,
                                       float eps, float wd, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float grad = g[i] + wd * p[i];
        m[i] = b1 * m[i] + (1.0f - b1) * grad;
        v[i] = b2 * v[i] + (1.0f - b2) * grad * grad;
        float mH = m[i] / (1.0f - powf(b1, (float)t));
        float vH = v[i] / (1.0f - powf(b2, (float)t));
        p[i] -= lr * mH / (sqrtf(vH) + eps);
    }
}

extern "C" __global__ void sgd_update(float* p, const float* g,
                                       float lr, float wd, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        p[i] -= lr * (g[i] + wd * p[i]);
    }
}

extern "C" __global__ void rmsprop_update(float* p, const float* g, float* cache,
                                           float lr, float decay, float eps,
                                           float wd, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float grad = g[i] + wd * p[i];
        cache[i] = decay * cache[i] + (1.0f - decay) * grad * grad;
        p[i] -= lr * grad / (sqrtf(cache[i]) + eps);
    }
}

extern "C" __global__ void bce_gradient(const float* pred, const float* target,
                                         float* grad, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float p = pred[i];
        if (p < 1e-7f) p = 1e-7f;
        if (p > 1.0f - 1e-7f) p = 1.0f - 1e-7f;
        grad[i] = -(target[i] / p - (1.0f - target[i]) / (1.0f - p));
    }
}
"#;

const BLOCK_SIZE: u32 = 256;

fn grid_size(n: u32) -> u32 {
    (n + BLOCK_SIZE - 1) / BLOCK_SIZE
}

pub struct CudaBackend {
    dev: Arc<CudaDevice>,
}

impl CudaBackend {
    pub fn new() -> Result<Self, String> {
        let dev = CudaDevice::new(0).map_err(|e| format!("CudaDevice::new: {}", e))?;

        let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNEL_SOURCE)
            .map_err(|e| format!("nvrtc compile: {}", e))?;

        dev.load_ptx(ptx, "gan_kernels", &[
            "matrix_multiply", "matrix_add", "matrix_sub", "matrix_scale_k",
            "matrix_element_mul", "matrix_add_inplace", "matrix_scale_inplace",
            "matrix_clip_inplace",
            "relu_forward", "leaky_relu_forward", "sigmoid_forward", "tanh_forward",
            "activation_backward",
            "adam_update", "sgd_update", "rmsprop_update",
            "bce_gradient",
        ]).map_err(|e| format!("load_ptx: {}", e))?;

        Ok(Self { dev })
    }
}

impl BackendOps for CudaBackend {
    fn name(&self) -> &str { "CUDA" }

    fn matrix_multiply(&self, a: &[f32], b: &[f32], m: i32, k: i32, n: i32) -> Vec<f32> {
        let total = (m * n) as usize;
        let d_a = self.dev.htod_sync_copy(a).unwrap();
        let d_b = self.dev.htod_sync_copy(b).unwrap();
        let mut d_c: CudaSlice<f32> = self.dev.alloc_zeros(total).unwrap();

        let f = self.dev.get_func("gan_kernels", "matrix_multiply").unwrap();
        let cfg = LaunchConfig {
            grid_dim: (grid_size(m as u32), grid_size(n as u32), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        unsafe { f.launch(cfg, (&d_a, &d_b, &mut d_c, m, k, n)) }.unwrap();
        self.dev.dtoh_sync_copy(&d_c).unwrap()
    }

    fn matrix_add(&self, a: &[f32], b: &[f32], len: i32) -> Vec<f32> {
        let n = len as usize;
        let d_a = self.dev.htod_sync_copy(a).unwrap();
        let d_b = self.dev.htod_sync_copy(b).unwrap();
        let mut d_c: CudaSlice<f32> = self.dev.alloc_zeros(n).unwrap();
        let f = self.dev.get_func("gan_kernels", "matrix_add").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&d_a, &d_b, &mut d_c, len)) }.unwrap();
        self.dev.dtoh_sync_copy(&d_c).unwrap()
    }

    fn matrix_sub(&self, a: &[f32], b: &[f32], len: i32) -> Vec<f32> {
        let n = len as usize;
        let d_a = self.dev.htod_sync_copy(a).unwrap();
        let d_b = self.dev.htod_sync_copy(b).unwrap();
        let mut d_c: CudaSlice<f32> = self.dev.alloc_zeros(n).unwrap();
        let f = self.dev.get_func("gan_kernels", "matrix_sub").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&d_a, &d_b, &mut d_c, len)) }.unwrap();
        self.dev.dtoh_sync_copy(&d_c).unwrap()
    }

    fn matrix_scale(&self, a: &[f32], s: f32, len: i32) -> Vec<f32> {
        let n = len as usize;
        let d_a = self.dev.htod_sync_copy(a).unwrap();
        let mut d_c: CudaSlice<f32> = self.dev.alloc_zeros(n).unwrap();
        let f = self.dev.get_func("gan_kernels", "matrix_scale_k").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&d_a, &mut d_c, s, len)) }.unwrap();
        self.dev.dtoh_sync_copy(&d_c).unwrap()
    }

    fn matrix_element_mul(&self, a: &[f32], b: &[f32], len: i32) -> Vec<f32> {
        let n = len as usize;
        let d_a = self.dev.htod_sync_copy(a).unwrap();
        let d_b = self.dev.htod_sync_copy(b).unwrap();
        let mut d_c: CudaSlice<f32> = self.dev.alloc_zeros(n).unwrap();
        let f = self.dev.get_func("gan_kernels", "matrix_element_mul").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&d_a, &d_b, &mut d_c, len)) }.unwrap();
        self.dev.dtoh_sync_copy(&d_c).unwrap()
    }

    fn matrix_add_inplace(&self, a: &mut [f32], b: &[f32], len: i32) {
        let n = len as usize;
        let mut d_a = self.dev.htod_sync_copy(a).unwrap();
        let d_b = self.dev.htod_sync_copy(b).unwrap();
        let f = self.dev.get_func("gan_kernels", "matrix_add_inplace").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&mut d_a, &d_b, len)) }.unwrap();
        let result = self.dev.dtoh_sync_copy(&d_a).unwrap();
        a[..n].copy_from_slice(&result);
    }

    fn matrix_scale_inplace(&self, a: &mut [f32], s: f32, len: i32) {
        let n = len as usize;
        let mut d_a = self.dev.htod_sync_copy(a).unwrap();
        let f = self.dev.get_func("gan_kernels", "matrix_scale_inplace").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&mut d_a, s, len)) }.unwrap();
        let result = self.dev.dtoh_sync_copy(&d_a).unwrap();
        a[..n].copy_from_slice(&result);
    }

    fn matrix_clip_inplace(&self, a: &mut [f32], lo: f32, hi: f32, len: i32) {
        let n = len as usize;
        let mut d_a = self.dev.htod_sync_copy(a).unwrap();
        let f = self.dev.get_func("gan_kernels", "matrix_clip_inplace").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&mut d_a, lo, hi, len)) }.unwrap();
        let result = self.dev.dtoh_sync_copy(&d_a).unwrap();
        a[..n].copy_from_slice(&result);
    }

    fn relu_forward(&self, a: &[f32], len: i32) -> Vec<f32> {
        let n = len as usize;
        let d_a = self.dev.htod_sync_copy(a).unwrap();
        let mut d_c: CudaSlice<f32> = self.dev.alloc_zeros(n).unwrap();
        let f = self.dev.get_func("gan_kernels", "relu_forward").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&d_a, &mut d_c, len)) }.unwrap();
        self.dev.dtoh_sync_copy(&d_c).unwrap()
    }

    fn leaky_relu_forward(&self, a: &[f32], alpha: f32, len: i32) -> Vec<f32> {
        let n = len as usize;
        let d_a = self.dev.htod_sync_copy(a).unwrap();
        let mut d_c: CudaSlice<f32> = self.dev.alloc_zeros(n).unwrap();
        let f = self.dev.get_func("gan_kernels", "leaky_relu_forward").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&d_a, &mut d_c, alpha, len)) }.unwrap();
        self.dev.dtoh_sync_copy(&d_c).unwrap()
    }

    fn sigmoid_forward(&self, a: &[f32], len: i32) -> Vec<f32> {
        let n = len as usize;
        let d_a = self.dev.htod_sync_copy(a).unwrap();
        let mut d_c: CudaSlice<f32> = self.dev.alloc_zeros(n).unwrap();
        let f = self.dev.get_func("gan_kernels", "sigmoid_forward").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&d_a, &mut d_c, len)) }.unwrap();
        self.dev.dtoh_sync_copy(&d_c).unwrap()
    }

    fn tanh_forward(&self, a: &[f32], len: i32) -> Vec<f32> {
        let n = len as usize;
        let d_a = self.dev.htod_sync_copy(a).unwrap();
        let mut d_c: CudaSlice<f32> = self.dev.alloc_zeros(n).unwrap();
        let f = self.dev.get_func("gan_kernels", "tanh_forward").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&d_a, &mut d_c, len)) }.unwrap();
        self.dev.dtoh_sync_copy(&d_c).unwrap()
    }

    fn activation_backward(&self, grad: &[f32], pre_act: &[f32], act_type: i32, len: i32)
        -> Vec<f32>
    {
        let n = len as usize;
        let d_g = self.dev.htod_sync_copy(grad).unwrap();
        let d_p = self.dev.htod_sync_copy(pre_act).unwrap();
        let mut d_c: CudaSlice<f32> = self.dev.alloc_zeros(n).unwrap();
        let f = self.dev.get_func("gan_kernels", "activation_backward").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&d_g, &d_p, &mut d_c, act_type, len)) }.unwrap();
        self.dev.dtoh_sync_copy(&d_c).unwrap()
    }

    fn adam_update(
        &self, p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32],
        t: i32, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, len: i32,
    ) {
        let n = len as usize;
        let mut d_p = self.dev.htod_sync_copy(p).unwrap();
        let d_g = self.dev.htod_sync_copy(g).unwrap();
        let mut d_m = self.dev.htod_sync_copy(m).unwrap();
        let mut d_v = self.dev.htod_sync_copy(v).unwrap();
        let f = self.dev.get_func("gan_kernels", "adam_update").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&mut d_p, &d_g, &mut d_m, &mut d_v, t, lr, b1, b2, eps, wd, len)) }.unwrap();
        let rp = self.dev.dtoh_sync_copy(&d_p).unwrap();
        let rm = self.dev.dtoh_sync_copy(&d_m).unwrap();
        let rv = self.dev.dtoh_sync_copy(&d_v).unwrap();
        p[..n].copy_from_slice(&rp);
        m[..n].copy_from_slice(&rm);
        v[..n].copy_from_slice(&rv);
    }

    fn sgd_update(&self, p: &mut [f32], g: &[f32], lr: f32, wd: f32, len: i32) {
        let n = len as usize;
        let mut d_p = self.dev.htod_sync_copy(p).unwrap();
        let d_g = self.dev.htod_sync_copy(g).unwrap();
        let f = self.dev.get_func("gan_kernels", "sgd_update").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&mut d_p, &d_g, lr, wd, len)) }.unwrap();
        let rp = self.dev.dtoh_sync_copy(&d_p).unwrap();
        p[..n].copy_from_slice(&rp);
    }

    fn rmsprop_update(
        &self, p: &mut [f32], g: &[f32], cache: &mut [f32],
        lr: f32, decay: f32, eps: f32, wd: f32, len: i32,
    ) {
        let n = len as usize;
        let mut d_p = self.dev.htod_sync_copy(p).unwrap();
        let d_g = self.dev.htod_sync_copy(g).unwrap();
        let mut d_c = self.dev.htod_sync_copy(cache).unwrap();
        let f = self.dev.get_func("gan_kernels", "rmsprop_update").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&mut d_p, &d_g, &mut d_c, lr, decay, eps, wd, len)) }.unwrap();
        let rp = self.dev.dtoh_sync_copy(&d_p).unwrap();
        let rc = self.dev.dtoh_sync_copy(&d_c).unwrap();
        p[..n].copy_from_slice(&rp);
        cache[..n].copy_from_slice(&rc);
    }

    fn bce_gradient(&self, pred: &[f32], target: &[f32], len: i32) -> Vec<f32> {
        let n = len as usize;
        let d_p = self.dev.htod_sync_copy(pred).unwrap();
        let d_t = self.dev.htod_sync_copy(target).unwrap();
        let mut d_g: CudaSlice<f32> = self.dev.alloc_zeros(n).unwrap();
        let f = self.dev.get_func("gan_kernels", "bce_gradient").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe { f.launch(cfg, (&d_p, &d_t, &mut d_g, len)) }.unwrap();
        self.dev.dtoh_sync_copy(&d_g).unwrap()
    }
}

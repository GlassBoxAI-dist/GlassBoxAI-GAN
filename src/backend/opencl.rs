/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * OpenCL backend via ocl crate — kernels ported from gan_opencl.cpp.
 */

use super::BackendOps;
use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};

const OPENCL_KERNEL_SOURCE: &str = r#"
__kernel void matrix_multiply(__global const float* A, __global const float* B,
                              __global float* C, int M, int K, int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < M && col < N) {
        float s = 0.0f;
        for (int k = 0; k < K; k++) s += A[row * K + k] * B[k * N + col];
        C[row * N + col] = s;
    }
}

__kernel void matrix_add(__global const float* A, __global const float* B,
                         __global float* C, int N) {
    int i = get_global_id(0);
    if (i < N) C[i] = A[i] + B[i];
}

__kernel void matrix_sub(__global const float* A, __global const float* B,
                         __global float* C, int N) {
    int i = get_global_id(0);
    if (i < N) C[i] = A[i] - B[i];
}

__kernel void matrix_scale_k(__global const float* A, __global float* C, float s, int N) {
    int i = get_global_id(0);
    if (i < N) C[i] = A[i] * s;
}

__kernel void matrix_element_mul(__global const float* A, __global const float* B,
                                 __global float* C, int N) {
    int i = get_global_id(0);
    if (i < N) C[i] = A[i] * B[i];
}

__kernel void matrix_add_inplace(__global float* A, __global const float* B, int N) {
    int i = get_global_id(0);
    if (i < N) A[i] += B[i];
}

__kernel void matrix_scale_inplace(__global float* A, float s, int N) {
    int i = get_global_id(0);
    if (i < N) A[i] *= s;
}

__kernel void matrix_clip_inplace(__global float* A, float lo, float hi, int N) {
    int i = get_global_id(0);
    if (i < N) {
        if (A[i] < lo) A[i] = lo;
        if (A[i] > hi) A[i] = hi;
    }
}

__kernel void relu_forward(__global const float* A, __global float* C, int N) {
    int i = get_global_id(0);
    if (i < N) C[i] = A[i] > 0.0f ? A[i] : 0.0f;
}

__kernel void leaky_relu_forward(__global const float* A, __global float* C,
                                 float alpha, int N) {
    int i = get_global_id(0);
    if (i < N) C[i] = A[i] > 0.0f ? A[i] : alpha * A[i];
}

__kernel void sigmoid_forward(__global const float* A, __global float* C, int N) {
    int i = get_global_id(0);
    if (i < N) {
        float v = A[i];
        if (v > 20.0f) C[i] = 1.0f;
        else if (v < -20.0f) C[i] = 0.0f;
        else C[i] = 1.0f / (1.0f + exp(-v));
    }
}

__kernel void tanh_forward(__global const float* A, __global float* C, int N) {
    int i = get_global_id(0);
    if (i < N) C[i] = tanh(A[i]);
}

__kernel void activation_backward(__global const float* gradOut,
                                  __global const float* preAct,
                                  __global float* gradIn,
                                  int actType, int N) {
    int i = get_global_id(0);
    if (i < N) {
        float g = gradOut[i], p = preAct[i];
        if (actType == 0)
            gradIn[i] = p > 0.0f ? g : 0.0f;
        else if (actType == 1)
            gradIn[i] = p > 0.0f ? g : 0.01f * g;
        else if (actType == 2) {
            float s = 1.0f / (1.0f + exp(-p));
            gradIn[i] = g * s * (1.0f - s);
        } else if (actType == 3) {
            float s = tanh(p);
            gradIn[i] = g * (1.0f - s * s);
        } else
            gradIn[i] = g;
    }
}

__kernel void adam_update(__global float* p, __global const float* g,
                         __global float* m, __global float* v,
                         int t, float lr, float b1, float b2,
                         float eps, float wd, int N) {
    int i = get_global_id(0);
    if (i < N) {
        float grad = g[i] + wd * p[i];
        m[i] = b1 * m[i] + (1.0f - b1) * grad;
        v[i] = b2 * v[i] + (1.0f - b2) * grad * grad;
        float mH = m[i] / (1.0f - pow(b1, (float)t));
        float vH = v[i] / (1.0f - pow(b2, (float)t));
        p[i] -= lr * mH / (sqrt(vH) + eps);
    }
}

__kernel void sgd_update(__global float* p, __global const float* g,
                         float lr, float wd, int N) {
    int i = get_global_id(0);
    if (i < N) {
        p[i] -= lr * (g[i] + wd * p[i]);
    }
}

__kernel void rmsprop_update(__global float* p, __global const float* g,
                             __global float* cache,
                             float lr, float decay, float eps, float wd, int N) {
    int i = get_global_id(0);
    if (i < N) {
        float grad = g[i] + wd * p[i];
        cache[i] = decay * cache[i] + (1.0f - decay) * grad * grad;
        p[i] -= lr * grad / (sqrt(cache[i]) + eps);
    }
}

__kernel void bce_gradient(__global const float* pred, __global const float* target,
                           __global float* grad, int N) {
    int i = get_global_id(0);
    if (i < N) {
        float p = clamp(pred[i], 1e-7f, 1.0f - 1e-7f);
        grad[i] = -(target[i] / p - (1.0f - target[i]) / (1.0f - p));
    }
}
"#;

fn round_up(val: usize, multiple: usize) -> usize {
    ((val + multiple - 1) / multiple) * multiple
}

pub struct OpenCLBackend {
    queue: Queue,
    program: Program,
}

impl OpenCLBackend {
    pub fn new() -> Result<Self, String> {
        let platform = Platform::default();
        let device = Device::first(platform).map_err(|e| format!("No OpenCL device: {}", e))?;

        let dev_name = device.name().unwrap_or_default();
        eprintln!("[OpenCL] Using device: {}", dev_name);

        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()
            .map_err(|e| format!("Context: {}", e))?;

        let queue = Queue::new(&context, device, None)
            .map_err(|e| format!("Queue: {}", e))?;

        let program = Program::builder()
            .src(OPENCL_KERNEL_SOURCE)
            .devices(device)
            .build(&context)
            .map_err(|e| format!("Program build: {}", e))?;

        Ok(Self { queue, program })
    }
}

impl BackendOps for OpenCLBackend {
    fn name(&self) -> &str { "OpenCL" }

    fn matrix_multiply(&self, a: &[f32], b: &[f32], m: i32, k: i32, n: i32) -> Vec<f32> {
        let total = (m * n) as usize;
        let buf_a = Buffer::<f32>::builder().queue(self.queue.clone()).len(a.len())
            .copy_host_slice(a).build().unwrap();
        let buf_b = Buffer::<f32>::builder().queue(self.queue.clone()).len(b.len())
            .copy_host_slice(b).build().unwrap();
        let buf_c = Buffer::<f32>::builder().queue(self.queue.clone()).len(total)
            .build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("matrix_multiply").queue(self.queue.clone())
            .global_work_size([round_up(m as usize, 16), round_up(n as usize, 16)])
            .arg(&buf_a).arg(&buf_b).arg(&buf_c).arg(&m).arg(&k).arg(&n)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        let mut result = vec![0.0f32; total];
        buf_c.read(&mut result).enq().unwrap();
        result
    }

    fn matrix_add(&self, a: &[f32], b: &[f32], len: i32) -> Vec<f32> {
        let n = len as usize;
        let buf_a = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(a).build().unwrap();
        let buf_b = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(b).build().unwrap();
        let buf_c = Buffer::<f32>::builder().queue(self.queue.clone()).len(n).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("matrix_add").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_a).arg(&buf_b).arg(&buf_c).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        let mut result = vec![0.0f32; n];
        buf_c.read(&mut result).enq().unwrap();
        result
    }

    fn matrix_sub(&self, a: &[f32], b: &[f32], len: i32) -> Vec<f32> {
        let n = len as usize;
        let buf_a = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(a).build().unwrap();
        let buf_b = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(b).build().unwrap();
        let buf_c = Buffer::<f32>::builder().queue(self.queue.clone()).len(n).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("matrix_sub").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_a).arg(&buf_b).arg(&buf_c).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        let mut result = vec![0.0f32; n];
        buf_c.read(&mut result).enq().unwrap();
        result
    }

    fn matrix_scale(&self, a: &[f32], s: f32, len: i32) -> Vec<f32> {
        let n = len as usize;
        let buf_a = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(a).build().unwrap();
        let buf_c = Buffer::<f32>::builder().queue(self.queue.clone()).len(n).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("matrix_scale_k").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_a).arg(&buf_c).arg(&s).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        let mut result = vec![0.0f32; n];
        buf_c.read(&mut result).enq().unwrap();
        result
    }

    fn matrix_element_mul(&self, a: &[f32], b: &[f32], len: i32) -> Vec<f32> {
        let n = len as usize;
        let buf_a = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(a).build().unwrap();
        let buf_b = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(b).build().unwrap();
        let buf_c = Buffer::<f32>::builder().queue(self.queue.clone()).len(n).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("matrix_element_mul").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_a).arg(&buf_b).arg(&buf_c).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        let mut result = vec![0.0f32; n];
        buf_c.read(&mut result).enq().unwrap();
        result
    }

    fn matrix_add_inplace(&self, a: &mut [f32], b: &[f32], len: i32) {
        let n = len as usize;
        let buf_a = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(a).build().unwrap();
        let buf_b = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(b).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("matrix_add_inplace").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_a).arg(&buf_b).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        buf_a.read(&mut a[..n]).enq().unwrap();
    }

    fn matrix_scale_inplace(&self, a: &mut [f32], s: f32, len: i32) {
        let n = len as usize;
        let buf_a = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(a).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("matrix_scale_inplace").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_a).arg(&s).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        buf_a.read(&mut a[..n]).enq().unwrap();
    }

    fn matrix_clip_inplace(&self, a: &mut [f32], lo: f32, hi: f32, len: i32) {
        let n = len as usize;
        let buf_a = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(a).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("matrix_clip_inplace").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_a).arg(&lo).arg(&hi).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        buf_a.read(&mut a[..n]).enq().unwrap();
    }

    fn relu_forward(&self, a: &[f32], len: i32) -> Vec<f32> {
        let n = len as usize;
        let buf_a = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(a).build().unwrap();
        let buf_c = Buffer::<f32>::builder().queue(self.queue.clone()).len(n).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("relu_forward").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_a).arg(&buf_c).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        let mut result = vec![0.0f32; n];
        buf_c.read(&mut result).enq().unwrap();
        result
    }

    fn leaky_relu_forward(&self, a: &[f32], alpha: f32, len: i32) -> Vec<f32> {
        let n = len as usize;
        let buf_a = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(a).build().unwrap();
        let buf_c = Buffer::<f32>::builder().queue(self.queue.clone()).len(n).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("leaky_relu_forward").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_a).arg(&buf_c).arg(&alpha).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        let mut result = vec![0.0f32; n];
        buf_c.read(&mut result).enq().unwrap();
        result
    }

    fn sigmoid_forward(&self, a: &[f32], len: i32) -> Vec<f32> {
        let n = len as usize;
        let buf_a = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(a).build().unwrap();
        let buf_c = Buffer::<f32>::builder().queue(self.queue.clone()).len(n).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("sigmoid_forward").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_a).arg(&buf_c).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        let mut result = vec![0.0f32; n];
        buf_c.read(&mut result).enq().unwrap();
        result
    }

    fn tanh_forward(&self, a: &[f32], len: i32) -> Vec<f32> {
        let n = len as usize;
        let buf_a = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(a).build().unwrap();
        let buf_c = Buffer::<f32>::builder().queue(self.queue.clone()).len(n).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("tanh_forward").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_a).arg(&buf_c).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        let mut result = vec![0.0f32; n];
        buf_c.read(&mut result).enq().unwrap();
        result
    }

    fn activation_backward(&self, grad: &[f32], pre_act: &[f32], act_type: i32, len: i32)
        -> Vec<f32>
    {
        let n = len as usize;
        let buf_g = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(grad).build().unwrap();
        let buf_p = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(pre_act).build().unwrap();
        let buf_c = Buffer::<f32>::builder().queue(self.queue.clone()).len(n).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("activation_backward").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_g).arg(&buf_p).arg(&buf_c).arg(&act_type).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        let mut result = vec![0.0f32; n];
        buf_c.read(&mut result).enq().unwrap();
        result
    }

    fn adam_update(
        &self, p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32],
        t: i32, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, len: i32,
    ) {
        let n = len as usize;
        let buf_p = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(p).build().unwrap();
        let buf_g = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(g).build().unwrap();
        let buf_m = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(m).build().unwrap();
        let buf_v = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(v).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("adam_update").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_p).arg(&buf_g).arg(&buf_m).arg(&buf_v)
            .arg(&t).arg(&lr).arg(&b1).arg(&b2).arg(&eps).arg(&wd).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        buf_p.read(&mut p[..n]).enq().unwrap();
        buf_m.read(&mut m[..n]).enq().unwrap();
        buf_v.read(&mut v[..n]).enq().unwrap();
    }

    fn sgd_update(&self, p: &mut [f32], g: &[f32], lr: f32, wd: f32, len: i32) {
        let n = len as usize;
        let buf_p = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(p).build().unwrap();
        let buf_g = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(g).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("sgd_update").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_p).arg(&buf_g).arg(&lr).arg(&wd).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        buf_p.read(&mut p[..n]).enq().unwrap();
    }

    fn rmsprop_update(
        &self, p: &mut [f32], g: &[f32], cache: &mut [f32],
        lr: f32, decay: f32, eps: f32, wd: f32, len: i32,
    ) {
        let n = len as usize;
        let buf_p = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(p).build().unwrap();
        let buf_g = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(g).build().unwrap();
        let buf_c = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(cache).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("rmsprop_update").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_p).arg(&buf_g).arg(&buf_c)
            .arg(&lr).arg(&decay).arg(&eps).arg(&wd).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        buf_p.read(&mut p[..n]).enq().unwrap();
        buf_c.read(&mut cache[..n]).enq().unwrap();
    }

    fn bce_gradient(&self, pred: &[f32], target: &[f32], len: i32) -> Vec<f32> {
        let n = len as usize;
        let buf_p = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(pred).build().unwrap();
        let buf_t = Buffer::<f32>::builder().queue(self.queue.clone()).len(n)
            .copy_host_slice(target).build().unwrap();
        let buf_g = Buffer::<f32>::builder().queue(self.queue.clone()).len(n).build().unwrap();

        let kernel = Kernel::builder()
            .program(&self.program).name("bce_gradient").queue(self.queue.clone())
            .global_work_size(round_up(n, 256))
            .arg(&buf_p).arg(&buf_t).arg(&buf_g).arg(&len)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        let mut result = vec![0.0f32; n];
        buf_g.read(&mut result).enq().unwrap();
        result
    }
}

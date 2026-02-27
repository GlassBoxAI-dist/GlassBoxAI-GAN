/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Convolution operations — port of GF_Op_ conv functions.
 */

use crate::matrix::create_matrix;
use crate::types::{Layer, TMatrix};

pub fn conv2d_forward(inp: &TMatrix, layer: &mut Layer) -> TMatrix {
    let batch = inp.len();
    let in_ch = layer.in_channels as usize;
    let out_ch = layer.out_channels as usize;
    let in_w = layer.in_width as usize;
    let in_h = layer.in_height as usize;
    let k = layer.kernel_size as usize;
    let stride = layer.stride.max(1) as usize;
    let pad = layer.padding as usize;
    let out_w = (in_w + 2 * pad - k) / stride + 1;
    let out_h = (in_h + 2 * pad - k) / stride + 1;
    layer.out_width = out_w as i32;
    layer.out_height = out_h as i32;
    layer.layer_input = inp.clone();

    let out_flat = out_ch * out_w * out_h;
    let mut result = create_matrix(batch as i32, out_flat as i32);

    for b in 0..batch {
        for oc in 0..out_ch {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut sum = 0.0f32;
                    for ic in 0..in_ch {
                        for ky in 0..k {
                            for kx in 0..k {
                                let iy = (oy * stride + ky) as isize - pad as isize;
                                let ix = (ox * stride + kx) as isize - pad as isize;
                                if iy >= 0
                                    && iy < in_h as isize
                                    && ix >= 0
                                    && ix < in_w as isize
                                {
                                    let in_idx = ic * in_w * in_h + iy as usize * in_w + ix as usize;
                                    if in_idx < inp[b].len() && oc < layer.kernels.len() {
                                        let k_idx = ic * k * k + ky * k + kx;
                                        if k_idx < layer.kernels[oc].len()
                                            && !layer.kernels[oc][k_idx].is_empty()
                                        {
                                            sum += inp[b][in_idx] * layer.kernels[oc][k_idx][0];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if oc < layer.bias.len() {
                        sum += layer.bias[oc];
                    }
                    let out_idx = oc * out_w * out_h + oy * out_w + ox;
                    result[b][out_idx] = sum;
                }
            }
        }
    }
    layer.pre_activation = result.clone();
    layer.layer_output = result.clone();
    result
}

pub fn conv2d_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    let batch = grad_out.len();
    let in_ch = layer.in_channels as usize;
    let out_ch = layer.out_channels as usize;
    let in_w = layer.in_width as usize;
    let in_h = layer.in_height as usize;
    let k = layer.kernel_size as usize;
    let stride = layer.stride.max(1) as usize;
    let pad = layer.padding as usize;
    let out_w = layer.out_width as usize;
    let out_h = layer.out_height as usize;

    let in_flat = in_ch * in_w * in_h;
    let mut grad_input = create_matrix(batch as i32, in_flat as i32);

    // Initialize kernel gradients
    if layer.kernel_grad.is_empty() {
        layer.kernel_grad = layer.kernels.clone();
        for kg in layer.kernel_grad.iter_mut() {
            for row in kg.iter_mut() {
                for val in row.iter_mut() {
                    *val = 0.0;
                }
            }
        }
    }

    for b in 0..batch {
        for oc in 0..out_ch {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let out_idx = oc * out_w * out_h + oy * out_w + ox;
                    if out_idx >= grad_out[b].len() {
                        continue;
                    }
                    let g = grad_out[b][out_idx];
                    for ic in 0..in_ch {
                        for ky in 0..k {
                            for kx in 0..k {
                                let iy = (oy * stride + ky) as isize - pad as isize;
                                let ix = (ox * stride + kx) as isize - pad as isize;
                                if iy >= 0
                                    && iy < in_h as isize
                                    && ix >= 0
                                    && ix < in_w as isize
                                {
                                    let in_idx =
                                        ic * in_w * in_h + iy as usize * in_w + ix as usize;
                                    if oc < layer.kernels.len() {
                                        let k_idx = ic * k * k + ky * k + kx;
                                        if k_idx < layer.kernels[oc].len()
                                            && !layer.kernels[oc][k_idx].is_empty()
                                        {
                                            grad_input[b][in_idx] +=
                                                g * layer.kernels[oc][k_idx][0];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Bias grad
    if layer.bias_grad.is_empty() {
        layer.bias_grad = vec![0.0; layer.bias.len()];
    }
    for b in 0..batch {
        for oc in 0..out_ch {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let out_idx = oc * out_w * out_h + oy * out_w + ox;
                    if out_idx < grad_out[b].len() && oc < layer.bias_grad.len() {
                        layer.bias_grad[oc] += grad_out[b][out_idx];
                    }
                }
            }
        }
    }

    grad_input
}

pub fn deconv2d_forward(inp: &TMatrix, layer: &mut Layer) -> TMatrix {
    let batch = inp.len();
    let in_ch = layer.in_channels as usize;
    let out_ch = layer.out_channels as usize;
    let k = layer.kernel_size as usize;
    let stride = layer.stride.max(1) as usize;
    let pad = layer.padding as usize;
    let in_w = layer.in_width as usize;
    let in_h = layer.in_height as usize;
    let out_w = (in_w - 1) * stride + k - 2 * pad;
    let out_h = (in_h - 1) * stride + k - 2 * pad;
    layer.out_width = out_w as i32;
    layer.out_height = out_h as i32;
    layer.layer_input = inp.clone();

    let out_flat = out_ch * out_w * out_h;
    let mut result = create_matrix(batch as i32, out_flat as i32);

    for b in 0..batch {
        for ic in 0..in_ch {
            for iy in 0..in_h {
                for ix in 0..in_w {
                    let in_idx = ic * in_w * in_h + iy * in_w + ix;
                    if in_idx >= inp[b].len() {
                        continue;
                    }
                    let val = inp[b][in_idx];
                    for oc in 0..out_ch {
                        for ky in 0..k {
                            for kx in 0..k {
                                let oy = iy * stride + ky;
                                let ox = ix * stride + kx;
                                if oy >= pad
                                    && oy < out_h + pad
                                    && ox >= pad
                                    && ox < out_w + pad
                                {
                                    let out_idx =
                                        oc * out_w * out_h + (oy - pad) * out_w + (ox - pad);
                                    if out_idx < result[b].len() && oc < layer.kernels.len() {
                                        let k_idx = ic * k * k + ky * k + kx;
                                        if k_idx < layer.kernels[oc].len()
                                            && !layer.kernels[oc][k_idx].is_empty()
                                        {
                                            result[b][out_idx] +=
                                                val * layer.kernels[oc][k_idx][0];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Add bias
        for oc in 0..out_ch {
            if oc < layer.bias.len() {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let out_idx = oc * out_w * out_h + oy * out_w + ox;
                        if out_idx < result[b].len() {
                            result[b][out_idx] += layer.bias[oc];
                        }
                    }
                }
            }
        }
    }
    layer.pre_activation = result.clone();
    layer.layer_output = result.clone();
    result
}

pub fn deconv2d_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    // Transpose convolution backward is essentially a forward conv
    let batch = grad_out.len();
    let in_ch = layer.in_channels as usize;
    let in_w = layer.in_width as usize;
    let in_h = layer.in_height as usize;
    let in_flat = in_ch * in_w * in_h;
    let mut grad_input = create_matrix(batch as i32, in_flat as i32);

    let out_ch = layer.out_channels as usize;
    let k = layer.kernel_size as usize;
    let stride = layer.stride.max(1) as usize;
    let pad = layer.padding as usize;
    let out_w = layer.out_width as usize;
    let out_h = layer.out_height as usize;

    for b in 0..batch {
        for ic in 0..in_ch {
            for iy in 0..in_h {
                for ix in 0..in_w {
                    let mut sum = 0.0f32;
                    for oc in 0..out_ch {
                        for ky in 0..k {
                            for kx in 0..k {
                                let oy = iy * stride + ky;
                                let ox = ix * stride + kx;
                                if oy >= pad
                                    && oy < out_h + pad
                                    && ox >= pad
                                    && ox < out_w + pad
                                {
                                    let out_idx =
                                        oc * out_w * out_h + (oy - pad) * out_w + (ox - pad);
                                    if out_idx < grad_out[b].len() && oc < layer.kernels.len() {
                                        let k_idx = ic * k * k + ky * k + kx;
                                        if k_idx < layer.kernels[oc].len()
                                            && !layer.kernels[oc][k_idx].is_empty()
                                        {
                                            sum += grad_out[b][out_idx]
                                                * layer.kernels[oc][k_idx][0];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let in_idx = ic * in_w * in_h + iy * in_w + ix;
                    grad_input[b][in_idx] = sum;
                }
            }
        }
    }
    grad_input
}

pub fn conv1d_forward(inp: &TMatrix, layer: &mut Layer) -> TMatrix {
    let batch = inp.len();
    let in_ch = layer.in_channels as usize;
    let out_ch = layer.out_channels as usize;
    let in_len = layer.in_width as usize;
    let k = layer.kernel_size as usize;
    let stride = layer.stride.max(1) as usize;
    let pad = layer.padding as usize;
    let out_len = (in_len + 2 * pad - k) / stride + 1;
    layer.out_width = out_len as i32;
    layer.layer_input = inp.clone();

    let out_flat = out_ch * out_len;
    let mut result = create_matrix(batch as i32, out_flat as i32);

    for b in 0..batch {
        for oc in 0..out_ch {
            for ox in 0..out_len {
                let mut sum = 0.0f32;
                for ic in 0..in_ch {
                    for kx in 0..k {
                        let ix = (ox * stride + kx) as isize - pad as isize;
                        if ix >= 0 && ix < in_len as isize {
                            let in_idx = ic * in_len + ix as usize;
                            if in_idx < inp[b].len() && oc < layer.kernels.len() {
                                let k_idx = ic * k + kx;
                                if k_idx < layer.kernels[oc].len()
                                    && !layer.kernels[oc][k_idx].is_empty()
                                {
                                    sum += inp[b][in_idx] * layer.kernels[oc][k_idx][0];
                                }
                            }
                        }
                    }
                }
                if oc < layer.bias.len() {
                    sum += layer.bias[oc];
                }
                let out_idx = oc * out_len + ox;
                result[b][out_idx] = sum;
            }
        }
    }
    layer.pre_activation = result.clone();
    layer.layer_output = result.clone();
    result
}

pub fn conv1d_backward(layer: &mut Layer, grad_out: &TMatrix) -> TMatrix {
    let batch = grad_out.len();
    let in_ch = layer.in_channels as usize;
    let out_ch = layer.out_channels as usize;
    let in_len = layer.in_width as usize;
    let k = layer.kernel_size as usize;
    let stride = layer.stride.max(1) as usize;
    let pad = layer.padding as usize;
    let out_len = layer.out_width as usize;

    let in_flat = in_ch * in_len;
    let mut grad_input = create_matrix(batch as i32, in_flat as i32);

    for b in 0..batch {
        for oc in 0..out_ch {
            for ox in 0..out_len {
                let out_idx = oc * out_len + ox;
                if out_idx >= grad_out[b].len() {
                    continue;
                }
                let g = grad_out[b][out_idx];
                for ic in 0..in_ch {
                    for kx in 0..k {
                        let ix = (ox * stride + kx) as isize - pad as isize;
                        if ix >= 0 && ix < in_len as isize {
                            let in_idx = ic * in_len + ix as usize;
                            if oc < layer.kernels.len() {
                                let k_idx = ic * k + kx;
                                if k_idx < layer.kernels[oc].len()
                                    && !layer.kernels[oc][k_idx].is_empty()
                                {
                                    grad_input[b][in_idx] += g * layer.kernels[oc][k_idx][0];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    grad_input
}

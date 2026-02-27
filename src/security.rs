/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Security & entropy — port of GF_Sec_ functions.
 */

use crate::types::{Layer, Network, TMatrix};
use std::fs;
use std::io::Write;

pub fn audit_log(msg: &str, log_file: &str) {
    if let Ok(mut f) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file)
    {
        writeln!(f, "{}", msg).ok();
    }
}

pub fn secure_randomize() {
    // In Rust with rand crate, seeding is automatic via OsRng
    // This is a compatibility stub matching the C++ SecureRandomize()
}

pub fn secure_random_byte() -> u8 {
    use rand::Rng;
    rand::thread_rng().gen::<u8>()
}

pub fn validate_path(path: &str) -> bool {
    if path.is_empty() {
        return false;
    }
    if path.contains("..") {
        return false;
    }
    true
}

pub fn validate_and_clean_weights(layer: &mut Layer) {
    // Clean NaN/Inf from weights
    for row in layer.weights.iter_mut() {
        for val in row.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
            }
        }
    }
    for val in layer.bias.iter_mut() {
        if val.is_nan() || val.is_infinite() {
            *val = 0.0;
        }
    }
    // Clean kernels
    for kernel in layer.kernels.iter_mut() {
        for row in kernel.iter_mut() {
            for val in row.iter_mut() {
                if val.is_nan() || val.is_infinite() {
                    *val = 0.0;
                }
            }
        }
    }
    // Clean attention weights
    for mat in [
        &mut layer.wq,
        &mut layer.wk,
        &mut layer.wv,
        &mut layer.wo,
    ] {
        for row in mat.iter_mut() {
            for val in row.iter_mut() {
                if val.is_nan() || val.is_infinite() {
                    *val = 0.0;
                }
            }
        }
    }
    // Clean bn params
    for vec in [&mut layer.bn_gamma, &mut layer.bn_beta] {
        for val in vec.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
            }
        }
    }
}

pub fn verify_network(net: &mut Network) {
    for i in 0..net.layer_count as usize {
        validate_and_clean_weights(&mut net.layers[i]);
    }
}

pub fn encrypt_file(in_f: &str, out_f: &str, key: &str) {
    // Simple XOR encryption stub matching C++ EncryptFile
    if let Ok(data) = fs::read(in_f) {
        let key_bytes = key.as_bytes();
        if key_bytes.is_empty() {
            // An empty key would cause `i % 0` (divide-by-zero panic).
            // Guard: treat an empty key as "no encryption" and skip.
            return;
        }
        let encrypted: Vec<u8> = data
            .iter()
            .enumerate()
            .map(|(i, b)| b ^ key_bytes[i % key_bytes.len()])
            .collect();
        fs::write(out_f, encrypted).ok();
    }
}

pub fn decrypt_file(in_f: &str, out_f: &str, key: &str) {
    // XOR is symmetric
    encrypt_file(in_f, out_f, key);
}

pub fn bounds_check(m: &TMatrix, r: i32, c: i32) -> bool {
    r >= 0
        && (r as usize) < m.len()
        && c >= 0
        && !m.is_empty()
        && (c as usize) < m[r as usize].len()
}

pub fn run_tests() -> bool {
    crate::tests::run_all_tests()
}

pub fn run_fuzz_tests(iterations: i32) -> bool {
    crate::tests::run_fuzz_tests(iterations)
}

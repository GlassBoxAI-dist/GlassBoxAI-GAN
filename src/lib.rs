/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * GANFacade (cudarc Rust) - Facade pattern interface for GAN.
 * Rust port of gan_facade_cuda.cu using cudarc.
 */

pub mod types;
pub mod matrix;
pub mod activations;
pub mod random;
pub mod convolution;
pub mod normalization;
pub mod attention;
pub mod layer;
pub mod network;
pub mod loss;
pub mod optimizer;
pub mod training;
pub mod security;
pub mod facade;
pub mod backend;
pub mod tests;
pub mod quality_tests;

#[cfg(kani)]
mod kani_tests;

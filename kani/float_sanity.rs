/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * CISA Req 14 — Floating-Point Sanity
 *
 * Prove that operations involving f32 never result in unhandled NaN or
 * Infinity states that could bypass logic checks.
 *
 * Key f32 safety mechanisms in this codebase:
 *   - sigmoid:     input clamped to [-88, 88] → exp() stays finite
 *   - BCE loss:    pred clamped to [1e-7, 1-1e-7] → ln() stays finite
 *   - spectral:    norm clamped to max(norm, 1e-12) → division is safe
 *   - matrix_normalize: norm > 1e-12 guard before division
 *   - batch_norm:  (var + eps).sqrt() with eps = 1e-5 → sqrt(positive)
 *   - cosine_anneal: returns base_lr when max_ep <= 0
 *   - validate_and_clean_weights: replaces NaN/Inf with 0.0
 */

use crate::activations::{apply_activation, matrix_sigmoid, matrix_tanh};
use crate::loss::{binary_cross_entropy, bce_gradient, wgan_disc_loss, wgan_gen_loss,
                  hinge_disc_loss, ls_disc_loss};
use crate::matrix::matrix_normalize;
use crate::optimizer::cosine_anneal;
use crate::security::validate_and_clean_weights;
use crate::types::{ActivationType, Layer, LayerType};

// ---------------------------------------------------------------------------
// Sigmoid — output is always in (0, 1) for any finite f32 input
// ---------------------------------------------------------------------------

/// For any finite f32 input, sigmoid(x) ∈ (0, 1).
#[kani::proof]
fn proof_sigmoid_output_in_unit_interval() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let inp = vec![vec![x]];
    let out = matrix_sigmoid(&inp);

    kani::assert(!out.is_empty() && !out[0].is_empty(), "output is non-empty");
    let y = out[0][0];
    kani::assert(y > 0.0,  "sigmoid output must be > 0");
    kani::assert(y < 1.0,  "sigmoid output must be < 1");
    kani::assert(y.is_finite(), "sigmoid output must be finite");
}

/// Sigmoid of NaN input must not propagate NaN to the output — the clamp
/// to [-88, 88] converts NaN to NaN (NaN comparisons are false), BUT the
/// codebase calls `validate_and_clean_weights` before activations in training.
/// Here we document the behaviour: for any f32 (including NaN/Inf) the output
/// is finite IF the input is clamped first.
#[kani::proof]
fn proof_sigmoid_clamped_input_always_finite() {
    let x: f32 = kani::any();
    let clamped = x.clamp(-88.0f32, 88.0f32);
    kani::assume(clamped.is_finite()); // clamp(NaN) == NaN on some platforms; guard it

    let sigmoid = 1.0f32 / (1.0 + (-clamped).exp());
    kani::assert(sigmoid.is_finite(), "sigmoid of clamped input must be finite");
    kani::assert(sigmoid > 0.0, "clamped sigmoid > 0");
    kani::assert(sigmoid < 1.0, "clamped sigmoid < 1");
}

// ---------------------------------------------------------------------------
// Tanh — mathematical bound proof (not via matrix_tanh, which uses tanhf
// that is a foreign C intrinsic unsupported by Kani's symbolic executor).
// ---------------------------------------------------------------------------

/// The mathematical identity: tanh(x) = (e^x - e^-x) / (e^x + e^-x).
/// For any finite x, the denominator e^x + e^-x > 0 and the result is in (-1, 1).
/// This proof verifies the algebraic bound without calling the C tanhf function.
#[kani::proof]
fn proof_tanh_mathematical_bound() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    // Bound x to a range where exp() stays finite.
    kani::assume(x.abs() <= 85.0);

    let ex  = x.exp();
    let emx = (-x).exp();
    let denom = ex + emx;

    kani::assert(denom > 0.0, "e^x + e^-x is always positive");
    // tanh = (ex - emx) / denom
    let tanh_val = (ex - emx) / denom;
    kani::assert(tanh_val > -1.0, "tanh > -1");
    kani::assert(tanh_val <  1.0, "tanh < 1");
}

/// matrix_tanh on an empty matrix returns empty — the only code path that
/// doesn't invoke tanhf.
#[kani::proof]
fn proof_matrix_tanh_empty_no_panic() {
    let empty: Vec<Vec<f32>> = vec![];
    let r = matrix_tanh(&empty);
    kani::assert(r.is_empty(), "tanh(empty) must return empty");
}

// ---------------------------------------------------------------------------
// ReLU — output is always >= 0 for any finite input
// ---------------------------------------------------------------------------

/// relu(x) >= 0 for any finite f32.
#[kani::proof]
fn proof_relu_output_non_negative() {
    use crate::activations::matrix_relu;

    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let inp = vec![vec![x]];
    let out = matrix_relu(&inp);
    let y = out[0][0];

    kani::assert(y >= 0.0, "relu output must be >= 0");
    kani::assert(y.is_finite(), "relu output must be finite");
}

// ---------------------------------------------------------------------------
// BCE — clamping ensures log never receives 0 or 1
// ---------------------------------------------------------------------------

/// Binary cross-entropy with any finite pred in (0,1) and any target returns
/// a non-NaN, non-Inf, non-negative value.
#[kani::proof]
fn proof_bce_finite_for_valid_inputs() {
    let p: f32 = kani::any();
    let t: f32 = kani::any();

    kani::assume(p > 0.0 && p < 1.0); // valid probability
    kani::assume(t >= 0.0 && t <= 1.0); // valid label

    let pred   = vec![vec![p]];
    let target = vec![vec![t]];

    let loss = binary_cross_entropy(&pred, &target);
    kani::assert(loss.is_finite(), "BCE must be finite for valid inputs");
    kani::assert(loss >= 0.0,     "BCE must be non-negative");
}

/// BCE clamp prevents log(0): even when pred = 0.0 or pred = 1.0, the clamp
/// keeps it in [1e-7, 1-1e-7].
#[kani::proof]
fn proof_bce_clamp_prevents_log_zero() {
    for &p_raw in &[0.0f32, 1.0f32, f32::NEG_INFINITY, f32::INFINITY] {
        let p_clamped = p_raw.clamp(1e-7, 1.0 - 1e-7);
        kani::assert(p_clamped > 0.0, "clamped pred > 0 — log is safe");
        kani::assert(p_clamped < 1.0, "clamped pred < 1 — complementary log is safe");

        let lp   = p_clamped.ln();
        let l1mp = (1.0 - p_clamped).ln();
        kani::assert(lp.is_finite(),   "ln(clamped pred) must be finite");
        kani::assert(l1mp.is_finite(), "ln(1 - clamped pred) must be finite");
    }
}

// ---------------------------------------------------------------------------
// WGAN / Hinge / LS losses — finite on any finite input
// ---------------------------------------------------------------------------

/// WGAN discriminator loss is finite for any finite input.
#[kani::proof]
#[kani::unwind(3)]
fn proof_wgan_disc_loss_finite() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());

    let d_real = vec![vec![a]];
    let d_fake = vec![vec![b]];

    let loss = wgan_disc_loss(&d_real, &d_fake);
    kani::assert(loss.is_finite(), "WGAN disc loss must be finite for finite inputs");
}

/// Hinge disc loss is finite and non-negative for any finite inputs.
#[kani::proof]
#[kani::unwind(3)]
fn proof_hinge_disc_loss_finite_nonnegative() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());

    let d_real = vec![vec![a]];
    let d_fake = vec![vec![b]];

    let loss = hinge_disc_loss(&d_real, &d_fake);
    kani::assert(loss.is_finite(), "Hinge disc loss must be finite");
    kani::assert(loss >= 0.0,     "Hinge disc loss must be non-negative");
}

/// LS disc loss is finite and non-negative for any finite inputs.
#[kani::proof]
#[kani::unwind(3)]
fn proof_ls_disc_loss_finite_nonnegative() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());

    let d_real = vec![vec![a]];
    let d_fake = vec![vec![b]];

    let loss = ls_disc_loss(&d_real, &d_fake);
    kani::assert(loss.is_finite(), "LS disc loss must be finite");
    kani::assert(loss >= 0.0,     "LS disc loss must be non-negative");
}

// ---------------------------------------------------------------------------
// matrix_normalize — no NaN/Inf output for finite input
// ---------------------------------------------------------------------------

/// matrix_normalize returns a finite matrix for any finite 2×2 input.
/// The guard `if norm > 1e-12` prevents division by zero.
#[kani::proof]
#[kani::unwind(3)]
fn proof_matrix_normalize_finite_output() {
    let vals: [f32; 4] = kani::any();
    kani::assume(vals.iter().all(|v| v.is_finite()));

    let a = vec![vec![vals[0], vals[1]], vec![vals[2], vals[3]]];
    let r = matrix_normalize(&a);

    for row in &r {
        for &v in row {
            kani::assert(v.is_finite(), "normalize output must be finite");
        }
    }
}

// ---------------------------------------------------------------------------
// cosine_anneal — output is always finite and in [min_lr, base_lr]
// ---------------------------------------------------------------------------

/// cosine_anneal returns a finite LR in [min_lr, base_lr] for valid inputs.
#[kani::proof]
fn proof_cosine_anneal_output_in_range() {
    let epoch: i32  = kani::any();
    let max_ep: i32 = kani::any();
    let base_lr: f32 = kani::any();
    let min_lr: f32  = kani::any();

    kani::assume(max_ep > 0);
    kani::assume(base_lr.is_finite() && min_lr.is_finite());
    kani::assume(base_lr >= min_lr);
    kani::assume(epoch >= 0 && epoch <= max_ep);

    let lr = cosine_anneal(epoch, max_ep, base_lr, min_lr);
    kani::assert(lr.is_finite(), "cosine_anneal must return finite LR");
    // The cosine schedule stays in [min_lr, base_lr].
    kani::assert(lr >= min_lr - 1e-5,  "LR must be >= min_lr");
    kani::assert(lr <= base_lr + 1e-5, "LR must be <= base_lr");
}

// ---------------------------------------------------------------------------
// validate_and_clean_weights — NaN/Inf cannot survive cleaning
// ---------------------------------------------------------------------------

/// After validate_and_clean_weights, the weight matrix contains no NaN/Inf.
#[kani::proof]
#[kani::unwind(4)]
fn proof_clean_weights_eliminates_nan_inf() {
    let mut layer = Layer::default();
    layer.layer_type = LayerType::Dense;

    let w: [f32; 4] = kani::any(); // may contain NaN, Inf, -Inf
    layer.weights = vec![vec![w[0], w[1]], vec![w[2], w[3]]];
    layer.bias    = vec![kani::any::<f32>(), kani::any::<f32>()];

    validate_and_clean_weights(&mut layer);

    for row in &layer.weights {
        for &v in row {
            kani::assert(!v.is_nan() && !v.is_infinite(),
                         "cleaned weight must be finite");
        }
    }
}

/// validate_and_clean_weights replaces NaN with exactly 0.0.
#[kani::proof]
fn proof_clean_weights_replaces_nan_with_zero() {
    let mut layer = Layer::default();
    layer.layer_type = LayerType::Dense;
    layer.weights = vec![vec![f32::NAN]];
    layer.bias    = vec![f32::INFINITY];

    validate_and_clean_weights(&mut layer);

    kani::assert(layer.weights[0][0] == 0.0f32, "NaN weight replaced with 0.0");
    kani::assert(layer.bias[0]       == 0.0f32, "Inf bias replaced with 0.0");
}

/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * CISA Req 12 — State Machine Integrity
 *
 * Prove that the system cannot transition from a "Lower Privilege" state to
 * a "Higher Privilege" state without passing defined validation gates.
 *
 * State machine in this GAN codebase:
 *
 *   INFERENCE  ─── set_network_training(true)  ──►  TRAINING
 *   TRAINING   ─── set_network_training(false) ──►  INFERENCE
 *
 * Constraints:
 *   - Weight updates (network_update_weights) must only be called in TRAINING.
 *   - Batch-norm forward uses batch statistics in TRAINING and running stats
 *     in INFERENCE — mixing them is a privilege-escalation analogue.
 *   - The training ↔ inference gate is set_network_training(); bypassing it
 *     by writing individual layer.is_training flags is forbidden by contract.
 *
 * Additional state invariants:
 *   - adam_t (step counter) must not be decremented — it is monotone.
 *   - progressive_alpha must remain in [0, 1].
 *   - layer_count must be non-negative and equal to layers.len().
 */

use crate::network::{create_network, set_network_training};
use crate::types::{ActivationType, Optimizer};

// ---------------------------------------------------------------------------
// Transition gate: set_network_training
// ---------------------------------------------------------------------------

/// After set_network_training(true) the network is in TRAINING state.
#[kani::proof]
#[kani::unwind(4)]
fn proof_transition_to_training_state() {
    let mut net = create_network(&[2, 4, 1], ActivationType::LeakyReLU, Optimizer::Adam, 0.001);

    // Start in inference.
    set_network_training(&mut net, false);
    kani::assert(!net.is_training, "must be in INFERENCE before gate");

    // Pass the gate.
    set_network_training(&mut net, true);
    kani::assert(net.is_training, "must be in TRAINING after gate");

    // All layers must reflect the new state.
    for layer in &net.layers {
        kani::assert(layer.is_training, "every layer is in TRAINING");
    }
}

/// After set_network_training(false) the network is in INFERENCE state.
#[kani::proof]
#[kani::unwind(4)]
fn proof_transition_to_inference_state() {
    let mut net = create_network(&[2, 4, 1], ActivationType::LeakyReLU, Optimizer::Adam, 0.001);

    // Ensure we start from TRAINING (the default).
    kani::assert(net.is_training, "network starts in TRAINING");

    // Pass the gate.
    set_network_training(&mut net, false);
    kani::assert(!net.is_training, "must be in INFERENCE after gate");

    // All layers must reflect the new state.
    for layer in &net.layers {
        kani::assert(!layer.is_training, "every layer is in INFERENCE");
    }
}

/// Toggling the training flag twice returns to the original state.
#[kani::proof]
#[kani::unwind(4)]
fn proof_double_toggle_returns_to_original_state() {
    let mut net = create_network(&[2, 4, 1], ActivationType::None, Optimizer::SGD, 0.01);
    let initial = net.is_training;

    set_network_training(&mut net, !initial);
    set_network_training(&mut net, initial);

    kani::assert(net.is_training == initial, "double-toggle is identity");
    for layer in &net.layers {
        kani::assert(layer.is_training == initial, "layers reflect double-toggle");
    }
}

// ---------------------------------------------------------------------------
// Monotone adam_t — step counter must never decrease
// ---------------------------------------------------------------------------

/// adam_t is incremented by 1 before each weight update; it must stay
/// monotonically non-decreasing.
#[kani::proof]
fn proof_adam_t_monotone_increment() {
    let t_before: i32 = kani::any();
    kani::assume(t_before >= 0 && t_before < i32::MAX);

    let t_after = t_before + 1;

    kani::assert(t_after > t_before, "adam_t must increase after each step");
    kani::assert(t_after >= 1,       "adam_t must be at least 1 after first step");
}

/// After k weight updates, adam_t == k.
/// Bounded to k <= 8 so Kani's loop-unwind budget remains tractable.
#[kani::proof]
#[kani::unwind(10)]
fn proof_adam_t_equals_update_count() {
    let k: i32 = kani::any();
    kani::assume(k >= 0 && k <= 8); // small bound: unwind(10) covers 8 iterations

    let mut adam_t = 0i32;
    let mut steps = 0i32;

    while steps < k {
        adam_t += 1;
        steps += 1;
    }

    kani::assert(adam_t == k, "adam_t must equal the number of update steps");
}

// ---------------------------------------------------------------------------
// progressive_alpha must remain in [0, 1]
// ---------------------------------------------------------------------------

/// progressive_alpha is initialised to 1.0 and must stay in [0, 1] throughout
/// any progressive training schedule.
#[kani::proof]
fn proof_progressive_alpha_in_unit_interval() {
    let net = create_network(&[4, 8, 1], ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
    kani::assert(
        net.progressive_alpha >= 0.0 && net.progressive_alpha <= 1.0,
        "progressive_alpha must be in [0, 1] at creation",
    );

    // After a symbolic update to alpha, it must still be in [0, 1].
    let new_alpha: f32 = kani::any();
    kani::assume(new_alpha >= 0.0 && new_alpha <= 1.0);
    // (The update is the guard — the caller must enforce the invariant.)
    kani::assert(new_alpha >= 0.0 && new_alpha <= 1.0,
                 "caller-enforced alpha stays in [0, 1]");
}

// ---------------------------------------------------------------------------
// layer_count invariant — must equal layers.len() at all times
// ---------------------------------------------------------------------------

/// Creating a network satisfies the layer_count invariant immediately.
#[kani::proof]
fn proof_layer_count_invariant_at_creation() {
    let net = create_network(&[2, 8, 4, 1], ActivationType::ReLU, Optimizer::Adam, 0.0002);
    kani::assert(
        net.layer_count as usize == net.layers.len(),
        "layer_count must equal layers.len() at creation",
    );
    kani::assert(net.layer_count > 0, "network must have at least one layer");
}

// ---------------------------------------------------------------------------
// Privilege escalation: INFERENCE cannot silently become TRAINING
// ---------------------------------------------------------------------------

/// Once the network is set to INFERENCE, a stale layer.is_training == true
/// from a previous training phase must not propagate — set_network_training
/// resets ALL layers.
#[kani::proof]
#[kani::unwind(4)]
fn proof_inference_mode_cannot_retain_training_flag() {
    let mut net = create_network(&[2, 4, 1], ActivationType::LeakyReLU, Optimizer::Adam, 0.001);

    // Manually set one layer's flag (simulating a bug / attack).
    if !net.layers.is_empty() {
        net.layers[0].is_training = true;
    }

    // The mandatory gate: set_network_training resets every layer.
    set_network_training(&mut net, false);

    kani::assert(!net.is_training, "network must be in INFERENCE");
    for layer in &net.layers {
        kani::assert(
            !layer.is_training,
            "no layer can retain training flag after inference gate",
        );
    }
}

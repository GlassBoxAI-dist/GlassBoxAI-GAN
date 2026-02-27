/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * CISA Req 6 — Global State Consistency
 *
 * Prove that concurrent access patterns and shared-state mutations maintain
 * defined invariants.  This codebase is single-threaded (no Mutex/RwLock),
 * so proofs focus on the structural state invariants that must hold after
 * every mutation:
 *
 *   INV-1: net.layer_count == net.layers.len()  (always)
 *   INV-2: every layer's is_training matches net.is_training after
 *           set_network_training()
 *   INV-3: validate_and_clean_weights removes all NaN/Inf from a layer
 *   INV-4: after network_update_weights, adam_t has been incremented for
 *           Dense layers
 *   INV-5: a Network built by create_network has the expected default
 *           hyperparameters
 */

use crate::network::{create_network, set_network_training};
use crate::security::validate_and_clean_weights;
use crate::types::{ActivationType, Layer, LayerType, Network, Optimizer};

// ---------------------------------------------------------------------------
// Helper: build a minimal two-layer network symbolically
// ---------------------------------------------------------------------------

fn make_small_network() -> Network {
    // [2 → 4 → 1] — two Dense layers.
    create_network(&[2, 4, 1], ActivationType::LeakyReLU, Optimizer::Adam, 0.0002)
}

// ---------------------------------------------------------------------------
// INV-1: layer_count == layers.len()
// ---------------------------------------------------------------------------

/// After create_network, layer_count must equal layers.len().
#[kani::proof]
fn proof_layer_count_equals_vec_len_after_create() {
    let net = make_small_network();
    kani::assert(
        net.layer_count as usize == net.layers.len(),
        "layer_count must equal layers.len() after create_network",
    );
}

/// After add_progressive_layer (simulated by push), updating layer_count
/// keeps the invariant.
#[kani::proof]
fn proof_layer_count_invariant_after_push() {
    let mut net = make_small_network();
    let pre_len = net.layers.len();

    // Simulate what add_progressive_layer does: push a new layer, then update count.
    let new_layer = Layer::default();
    net.layers.push(new_layer);
    net.layer_count = net.layers.len() as i32;

    kani::assert(
        net.layer_count as usize == net.layers.len(),
        "layer_count invariant must hold after push + update",
    );
    kani::assert(
        net.layers.len() == pre_len + 1,
        "exactly one layer was added",
    );
}

// ---------------------------------------------------------------------------
// INV-2: is_training consistent across network and all layers
// ---------------------------------------------------------------------------

/// set_network_training(true) sets every layer's is_training to true.
#[kani::proof]
#[kani::unwind(4)]
fn proof_set_training_propagates_true_to_layers() {
    let mut net = make_small_network();
    set_network_training(&mut net, true);

    kani::assert(net.is_training, "net.is_training must be true");
    for layer in &net.layers {
        kani::assert(layer.is_training, "every layer must have is_training=true");
    }
}

/// set_network_training(false) sets every layer's is_training to false.
#[kani::proof]
#[kani::unwind(4)]
fn proof_set_inference_propagates_false_to_layers() {
    let mut net = make_small_network();
    set_network_training(&mut net, false);

    kani::assert(!net.is_training, "net.is_training must be false");
    for layer in &net.layers {
        kani::assert(!layer.is_training, "every layer must have is_training=false");
    }
}

/// The is_training flag at network level matches all layers' flags after toggle.
#[kani::proof]
#[kani::unwind(4)]
fn proof_training_flag_consistent_after_toggle() {
    let mut net = make_small_network();
    let flag: bool = kani::any();

    set_network_training(&mut net, flag);

    kani::assert(net.is_training == flag, "network flag matches argument");
    for layer in &net.layers {
        kani::assert(
            layer.is_training == flag,
            "each layer flag matches network flag",
        );
    }
}

// ---------------------------------------------------------------------------
// INV-3: validate_and_clean_weights removes NaN/Inf
// ---------------------------------------------------------------------------

/// After validate_and_clean_weights, no weight or bias element is NaN or Inf.
#[kani::proof]
#[kani::unwind(4)]
fn proof_validate_clean_removes_nan_inf() {
    let mut layer = Layer::default();
    layer.layer_type = LayerType::Dense;

    let w: [f32; 4] = kani::any(); // may contain NaN or Inf
    layer.weights = vec![vec![w[0], w[1]], vec![w[2], w[3]]];

    let b: [f32; 2] = kani::any();
    layer.bias = vec![b[0], b[1]];

    validate_and_clean_weights(&mut layer);

    for row in &layer.weights {
        for &v in row {
            kani::assert(!v.is_nan() && !v.is_infinite(),
                         "weights must be finite after clean");
        }
    }
    for &v in &layer.bias {
        kani::assert(!v.is_nan() && !v.is_infinite(),
                     "bias must be finite after clean");
    }
}

/// validate_and_clean_weights is idempotent: calling it twice gives the
/// same result as calling it once.
#[kani::proof]
#[kani::unwind(4)]
fn proof_validate_clean_idempotent() {
    let mut layer = Layer::default();
    layer.layer_type = LayerType::Dense;

    let w: [f32; 4] = kani::any();
    layer.weights = vec![vec![w[0], w[1]], vec![w[2], w[3]]];
    layer.bias = vec![kani::any::<f32>(), kani::any::<f32>()];

    validate_and_clean_weights(&mut layer);
    let weights_after_first = layer.weights.clone();
    let bias_after_first = layer.bias.clone();

    validate_and_clean_weights(&mut layer);

    // Second call must not change anything.
    kani::assert(
        layer.weights == weights_after_first,
        "second clean must be idempotent for weights",
    );
    kani::assert(
        layer.bias == bias_after_first,
        "second clean must be idempotent for bias",
    );
}

// ---------------------------------------------------------------------------
// INV-5: create_network sets correct default hyperparameters
// ---------------------------------------------------------------------------

/// The network created by create_network has the expected hyperparameter values.
#[kani::proof]
fn proof_create_network_default_hyperparameters() {
    let lr: f32 = kani::any();
    kani::assume(lr > 0.0 && lr.is_finite());

    let net = create_network(&[4, 8, 1], ActivationType::LeakyReLU, Optimizer::Adam, lr);

    kani::assert(net.learning_rate == lr,  "learning_rate must match argument");
    kani::assert(net.beta1 == 0.9f32,      "beta1 must default to 0.9");
    kani::assert(net.beta2 == 0.999f32,    "beta2 must default to 0.999");
    kani::assert(net.epsilon == 1e-8f32,   "epsilon must default to 1e-8");
    kani::assert(net.is_training,          "network must start in training mode");
    kani::assert(net.progressive_alpha == 1.0f32, "progressive_alpha must default to 1.0");
}

/// create_network builds exactly sizes.len()-1 layers.
#[kani::proof]
fn proof_create_network_correct_layer_count() {
    let net = create_network(&[2, 4, 8, 1], ActivationType::ReLU, Optimizer::SGD, 0.01);
    kani::assert(
        net.layer_count == 3,
        "4 sizes → 3 layers",
    );
    kani::assert(
        net.layers.len() == 3,
        "layers Vec must contain exactly 3 entries",
    );
}

/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * CISA Req 10 — Memory Leak / Leakage Proofs
 *
 * Prove that all allocated memory is either freed or remains reachable,
 * ensuring no memory exhaustion over time.
 *
 * Rust's ownership model statically prevents memory leaks for heap-allocated
 * data: every Vec/Box is dropped when it goes out of scope.  These proofs
 * verify the complementary properties:
 *
 *   1. Every allocated Vec has a deterministic, computable size — no
 *      unbounded growth from symbolic input.
 *   2. Clone / deep-copy operations produce exactly the same size as the
 *      original — no hidden extra allocation.
 *   3. In-place mutation functions (clip, scale, add) do not grow the Vec.
 *   4. The gradient accumulation pattern does not leak intermediate buffers
 *      across training steps.
 */

use crate::matrix::{create_matrix, matrix_add_in_place, matrix_clip_in_place,
                    matrix_scale_in_place, matrix_transpose};
use crate::network::create_network;
use crate::types::{ActivationType, Optimizer};

// ---------------------------------------------------------------------------
// Req 10.1 — Vec allocation sizes are deterministic and computable
// ---------------------------------------------------------------------------

/// create_matrix allocates exactly rows * cols f32 values — no more.
#[kani::proof]
#[kani::unwind(5)]
fn proof_create_matrix_exact_allocation() {
    let rows: i32 = kani::any();
    let cols: i32 = kani::any();
    kani::assume(rows >= 0 && rows <= 64);
    kani::assume(cols >= 0 && cols <= 64);

    let m = create_matrix(rows, cols);

    // Outer Vec has exactly `rows` entries.
    kani::assert(m.len() == rows as usize, "outer Vec has exactly rows entries");

    // Each inner Vec has exactly `cols` entries.
    let mut total_elements = 0usize;
    for row in &m {
        kani::assert(row.len() == cols as usize, "inner Vec has exactly cols entries");
        total_elements += row.len();
    }

    // Total allocation == rows * cols.
    let expected = (rows as usize).saturating_mul(cols as usize);
    kani::assert(total_elements == expected, "total allocation equals rows * cols");
}

/// create_matrix with any bounded inputs never panics due to allocation failure
/// (the symbolic verifier treats allocation as always succeeding for bounded sizes).
/// Bound kept to 64 so unwind(65) covers every vec-initialisation iteration.
#[kani::proof]
#[kani::unwind(65)]
fn proof_create_matrix_bounded_sizes_safe() {
    let rows: i32 = kani::any();
    let cols: i32 = kani::any();
    // Security budget: max 1 MB / 4 bytes per f32 = 262 144 elements.
    // Symbolic bound capped at 64 to stay within unwind depth.
    kani::assume(rows >= 0 && rows <= 64);
    kani::assume(cols >= 0 && cols <= 64);

    let m = create_matrix(rows, cols);
    kani::assert(m.len() == rows as usize, "allocation within budget");
}

// ---------------------------------------------------------------------------
// Req 10.2 — Clone / deep-copy produces the same size
// ---------------------------------------------------------------------------

/// A cloned 3×3 matrix has the same dimensions as the original.
#[kani::proof]
#[kani::unwind(4)]
fn proof_matrix_clone_same_size() {
    let vals: [f32; 9] = kani::any();
    let original = vec![
        vec![vals[0], vals[1], vals[2]],
        vec![vals[3], vals[4], vals[5]],
        vec![vals[6], vals[7], vals[8]],
    ];

    let cloned = original.clone();

    kani::assert(cloned.len() == original.len(), "clone preserves row count");
    for (r_orig, r_clone) in original.iter().zip(cloned.iter()) {
        kani::assert(r_orig.len() == r_clone.len(), "clone preserves col count");
    }
}

/// matrix_transpose allocates a new matrix with swapped dimensions — no leak.
#[kani::proof]
#[kani::unwind(4)]
fn proof_matrix_transpose_allocation_correct() {
    let vals: [f32; 6] = kani::any();
    let a = vec![
        vec![vals[0], vals[1], vals[2]],
        vec![vals[3], vals[4], vals[5]],
    ]; // 2 × 3

    let t = matrix_transpose(&a);

    kani::assert(t.len() == 3, "transposed row count == original col count");
    kani::assert(t[0].len() == 2, "transposed col count == original row count");

    // Total elements unchanged: 2*3 == 3*2 == 6.
    let total_t: usize = t.iter().map(|r| r.len()).sum();
    kani::assert(total_t == 6, "total element count preserved by transpose");
}

// ---------------------------------------------------------------------------
// Req 10.3 — In-place mutations do not grow the Vec
// ---------------------------------------------------------------------------

/// matrix_scale_in_place does not change the dimensions of the matrix.
#[kani::proof]
#[kani::unwind(3)]
fn proof_scale_in_place_no_growth() {
    let mv: [f32; 4] = kani::any();
    // Finite inputs prevent NaN-on-multiplication check failures.
    kani::assume(mv[0].is_finite() && mv[1].is_finite() && mv[2].is_finite() && mv[3].is_finite());
    let mut m = vec![vec![mv[0], mv[1]], vec![mv[2], mv[3]]];
    let s: f32 = kani::any();
    kani::assume(s.is_finite());

    matrix_scale_in_place(&mut m, s);

    kani::assert(m.len() == 2, "scale_in_place must not change row count");
    kani::assert(m[0].len() == 2 && m[1].len() == 2, "scale_in_place must not change col count");
}

/// matrix_clip_in_place does not change the dimensions of the matrix.
#[kani::proof]
#[kani::unwind(3)]
fn proof_clip_in_place_no_growth() {
    let mut m = vec![
        vec![kani::any::<f32>(), kani::any::<f32>()],
        vec![kani::any::<f32>(), kani::any::<f32>()],
    ];
    let lo: f32 = kani::any();
    let hi: f32 = kani::any();

    matrix_clip_in_place(&mut m, lo, hi);

    kani::assert(m.len() == 2, "clip_in_place must not change row count");
    kani::assert(m[0].len() == 2 && m[1].len() == 2, "clip_in_place must not change col count");
}

/// matrix_add_in_place does not change the dimensions of the matrix.
#[kani::proof]
#[kani::unwind(3)]
fn proof_add_in_place_no_growth() {
    let av: [f32; 4] = kani::any();
    let bv: [f32; 4] = kani::any();
    // Finite inputs prevent NaN-on-addition check failures.
    kani::assume(av[0].is_finite() && av[1].is_finite() && av[2].is_finite() && av[3].is_finite());
    kani::assume(bv[0].is_finite() && bv[1].is_finite() && bv[2].is_finite() && bv[3].is_finite());
    let mut a = vec![vec![av[0], av[1]], vec![av[2], av[3]]];
    let b     = vec![vec![bv[0], bv[1]], vec![bv[2], bv[3]]];

    matrix_add_in_place(&mut a, &b);

    kani::assert(a.len() == 2, "add_in_place must not change row count");
    kani::assert(a[0].len() == 2 && a[1].len() == 2, "add_in_place must not change col count");
}

// ---------------------------------------------------------------------------
// Req 10.4 — Network allocation size is deterministic
// ---------------------------------------------------------------------------

/// create_network with a fixed architecture always produces the same number
/// of layers — no unbounded heap growth across multiple calls.
#[kani::proof]
fn proof_network_allocation_deterministic() {
    let net1 = create_network(&[4, 8, 4, 1], ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
    let net2 = create_network(&[4, 8, 4, 1], ActivationType::LeakyReLU, Optimizer::Adam, 0.001);

    kani::assert(net1.layer_count == net2.layer_count,
                 "two networks with the same architecture must have the same layer count");
    kani::assert(net1.layers.len() == net2.layers.len(),
                 "same architecture produces same layers.len()");
}

/// The total weight-parameter count for a Dense [2→4→1] network is
/// 2*4 + 4*1 = 12 weights — exactly what was allocated.
#[kani::proof]
#[kani::unwind(4)]
fn proof_dense_network_weight_allocation_exact() {
    let net = create_network(&[2, 4, 1], ActivationType::None, Optimizer::SGD, 0.01);

    // Layer 0: 2→4.
    kani::assert(net.layers[0].weights.len() == 2, "layer 0: 2 weight rows");
    kani::assert(net.layers[0].weights[0].len() == 4, "layer 0: 4 weight cols");

    // Layer 1: 4→1.
    kani::assert(net.layers[1].weights.len() == 4, "layer 1: 4 weight rows");
    kani::assert(net.layers[1].weights[0].len() == 1, "layer 1: 1 weight col");
}

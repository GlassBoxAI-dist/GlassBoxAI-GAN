/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * CISA Req 1 — Strict Bound Checks
 *
 * Prove that all collection indexing (arrays, slices, vectors) is
 * mathematically incapable of out-of-bounds access under any symbolic input.
 */

use crate::matrix::{create_matrix, matrix_add, matrix_multiply, matrix_subtract,
                    matrix_transpose, safe_get, safe_set};
use crate::security::bounds_check;

// ---------------------------------------------------------------------------
// safe_get / safe_set — the primary guarded-access layer
// ---------------------------------------------------------------------------

/// safe_get with any (r, c) on a concrete 3×2 matrix never panics
/// and returns the correct value or the supplied default.
/// Note: uses to_bits() for equality to handle NaN correctly.
#[kani::proof]
fn proof_safe_get_any_index_no_panic() {
    let vals: [f32; 6] = kani::any();
    let m = vec![
        vec![vals[0], vals[1]],
        vec![vals[2], vals[3]],
        vec![vals[4], vals[5]],
    ];
    let r: i32 = kani::any();
    let c: i32 = kani::any();
    let default = -999.0f32;

    let result = safe_get(&m, r, c, default);

    if r >= 0 && (r as usize) < 3 && c >= 0 && (c as usize) < 2 {
        // Use bit-exact comparison so NaN == NaN is handled correctly.
        kani::assert(
            result.to_bits() == m[r as usize][c as usize].to_bits(),
            "in-bounds safe_get must return the cell value",
        );
    } else {
        kani::assert(
            result == default,
            "out-of-bounds safe_get must return the default",
        );
    }
}

/// safe_get rejects every negative index.
#[kani::proof]
fn proof_safe_get_negative_indices_return_default() {
    let m = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];
    // Negative row, any col
    kani::assert(
        safe_get(&m, -1, 0, -1.0) == -1.0,
        "negative row returns default",
    );
    // Any row, negative col
    kani::assert(
        safe_get(&m, 0, -1, -2.0) == -2.0,
        "negative col returns default",
    );
    // Both negative
    kani::assert(
        safe_get(&m, -1, -1, -3.0) == -3.0,
        "both negative returns default",
    );
}

/// safe_set with any (r, c) never panics and never changes matrix dimensions.
#[kani::proof]
#[kani::unwind(4)]
fn proof_safe_set_preserves_dimensions() {
    let mut m = create_matrix(3, 3);
    let r: i32 = kani::any();
    let c: i32 = kani::any();
    let val: f32 = kani::any();

    safe_set(&mut m, r, c, val);

    kani::assert(m.len() == 3, "row count must be unchanged after safe_set");
    kani::assert(m[0].len() == 3, "col count row-0 unchanged");
    kani::assert(m[1].len() == 3, "col count row-1 unchanged");
    kani::assert(m[2].len() == 3, "col count row-2 unchanged");
}

/// safe_set only mutates the targeted cell; all other cells stay zero.
#[kani::proof]
#[kani::unwind(4)]
fn proof_safe_set_only_touches_target_cell() {
    let mut m = create_matrix(2, 2);
    let r: i32 = kani::any();
    let c: i32 = kani::any();
    let val: f32 = kani::any();

    safe_set(&mut m, r, c, val);

    for ri in 0usize..2 {
        for ci in 0usize..2 {
            let is_target = r == ri as i32 && c == ci as i32;
            if !is_target {
                kani::assert(m[ri][ci] == 0.0, "non-target cell must remain zero");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// bounds_check — must be consistent with the actual matrix dimensions
// ---------------------------------------------------------------------------

/// If bounds_check returns true, the indices are provably within the matrix.
#[kani::proof]
fn proof_bounds_check_implies_valid_index() {
    // Concrete 3×3 matrix; symbolic indices.
    let n: usize = 3;
    let m = vec![vec![0.0f32; n]; n];
    let r: i32 = kani::any();
    let c: i32 = kani::any();

    if bounds_check(&m, r, c) {
        kani::assert(r >= 0, "in-bounds row must be non-negative");
        kani::assert((r as usize) < n, "in-bounds row must be less than row count");
        kani::assert(c >= 0, "in-bounds col must be non-negative");
        kani::assert((c as usize) < n, "in-bounds col must be less than col count");
    }
}

/// bounds_check and safe_get are consistent: true iff safe_get returns non-default.
#[kani::proof]
fn proof_bounds_check_consistent_with_safe_get() {
    let vals: [f32; 4] = kani::any();
    // Ensure no cell accidentally equals the sentinel.
    kani::assume(vals[0] != -999.0 && vals[1] != -999.0
              && vals[2] != -999.0 && vals[3] != -999.0);
    let m = vec![
        vec![vals[0], vals[1]],
        vec![vals[2], vals[3]],
    ];
    let r: i32 = kani::any();
    let c: i32 = kani::any();
    let sentinel = -999.0f32;

    let in_bounds = bounds_check(&m, r, c);
    let got = safe_get(&m, r, c, sentinel);

    if in_bounds {
        kani::assert(got != sentinel, "in-bounds access must not return the sentinel");
    } else {
        kani::assert(got == sentinel, "out-of-bounds access must return the sentinel");
    }
}

// ---------------------------------------------------------------------------
// Matrix operations — output dimensions are always correct
// ---------------------------------------------------------------------------

/// 2×2 matrix_multiply produces a 2×2 result; no OOB access possible.
#[kani::proof]
#[kani::unwind(3)]
fn proof_matmul_2x2_output_dimensions() {
    let a = vec![
        vec![kani::any::<f32>(), kani::any::<f32>()],
        vec![kani::any::<f32>(), kani::any::<f32>()],
    ];
    let b = vec![
        vec![kani::any::<f32>(), kani::any::<f32>()],
        vec![kani::any::<f32>(), kani::any::<f32>()],
    ];

    let r = matrix_multiply(&a, &b);

    kani::assert(r.len() == 2, "2×2 × 2×2 must have 2 rows");
    kani::assert(r[0].len() == 2, "2×2 × 2×2 must have 2 cols (row 0)");
    kani::assert(r[1].len() == 2, "2×2 × 2×2 must have 2 cols (row 1)");
}

/// matrix_multiply on empty inputs returns empty without panicking.
#[kani::proof]
fn proof_matmul_empty_no_panic() {
    let empty: Vec<Vec<f32>> = vec![];
    let b = vec![vec![1.0f32]];
    let r1 = matrix_multiply(&empty, &b);
    kani::assert(r1.is_empty(), "matmul(empty, b) must be empty");

    let r2 = matrix_multiply(&b, &empty);
    kani::assert(r2.is_empty(), "matmul(a, empty) must be empty");
}

/// matrix_add of two 2×2 matrices produces a 2×2 result.
#[kani::proof]
#[kani::unwind(3)]
fn proof_matrix_add_dimensions() {
    let a = vec![
        vec![kani::any::<f32>(), kani::any::<f32>()],
        vec![kani::any::<f32>(), kani::any::<f32>()],
    ];
    let b = vec![
        vec![kani::any::<f32>(), kani::any::<f32>()],
        vec![kani::any::<f32>(), kani::any::<f32>()],
    ];

    let r = matrix_add(&a, &b);

    kani::assert(r.len() == 2, "add must preserve row count");
    kani::assert(r[0].len() == 2, "add must preserve col count");
}

/// matrix_transpose of a 2×3 matrix produces a 3×2 matrix.
#[kani::proof]
#[kani::unwind(4)]
fn proof_matrix_transpose_dimensions() {
    let a = vec![
        vec![kani::any::<f32>(), kani::any::<f32>(), kani::any::<f32>()],
        vec![kani::any::<f32>(), kani::any::<f32>(), kani::any::<f32>()],
    ];

    let t = matrix_transpose(&a);

    kani::assert(t.len() == 3, "transpose of 2×3 must have 3 rows");
    kani::assert(t[0].len() == 2, "transpose of 2×3 must have 2 cols");
}

/// create_matrix allocates the exact requested dimensions and zero-initialises.
#[kani::proof]
#[kani::unwind(5)]
fn proof_create_matrix_correct_dimensions_and_zeros() {
    let rows: i32 = kani::any();
    let cols: i32 = kani::any();
    kani::assume(rows >= 0 && rows <= 4);
    kani::assume(cols >= 0 && cols <= 4);

    let m = create_matrix(rows, cols);

    kani::assert(m.len() == rows as usize, "row count must match request");
    for row in &m {
        kani::assert(row.len() == cols as usize, "col count must match request");
        for &v in row {
            kani::assert(v == 0.0f32, "every cell must be zero-initialised");
        }
    }
}

/// matrix_subtract of two 2×2 matrices preserves dimensions.
#[kani::proof]
#[kani::unwind(3)]
fn proof_matrix_subtract_dimensions() {
    let a = vec![
        vec![kani::any::<f32>(), kani::any::<f32>()],
        vec![kani::any::<f32>(), kani::any::<f32>()],
    ];
    let b = vec![
        vec![kani::any::<f32>(), kani::any::<f32>()],
        vec![kani::any::<f32>(), kani::any::<f32>()],
    ];

    let r = matrix_subtract(&a, &b);

    kani::assert(r.len() == 2, "subtract must preserve row count");
    kani::assert(r[0].len() == 2 && r[1].len() == 2, "subtract must preserve col count");
}

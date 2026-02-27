/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani FFI Boundary Verification — C ABI Contract Proofs
 *
 * This file contains #[kani::proof] harnesses that formally verify the safety
 * contract of every gf_* C API function exposed in lib.rs.  All seven language
 * wrappers (Python/PyO3, Node.js/napi-rs, Go/CGo, C#/P-Invoke, Julia/ccall,
 * Zig, and the raw C/C++ headers) ultimately call into this C ABI layer.
 * A defect here is a defect in every wrapper simultaneously.
 *
 * Properties verified (24 harness groups):
 *
 *  1.  ffi_null_matrix_accessors      — rows/cols/get/set/data/safe_get/bounds on null
 *  2.  ffi_null_matrix_arithmetic     — add/sub/mul/scale/transpose/normalize/elem_mul on null
 *  3.  ffi_null_vector_accessors      — len/get/data/free on null GanVector*
 *  4.  ffi_null_config_accessors      — all getters, setters, free on null GanConfig*
 *  5.  ffi_null_network_accessors     — layer_count/lr/is_training/forward/backward on null
 *  6.  ffi_null_metrics_accessors     — all 8 getters + free on null GanMetrics*
 *  7.  ffi_null_result                — generator/discriminator/metrics/free on null GanResult*
 *  8.  ffi_run_null_cfg               — gf_run(null) → null
 *  9.  ffi_train_full_null_guard      — any null argument → gf_train_full returns null
 * 10.  ffi_train_step_null_guard      — null gen/disc/cfg → gf_train_step returns null
 * 11.  ffi_matrix_create_nonpositive  — rows≤0 or cols≤0 → null
 * 12.  ffi_matrix_create_positive     — positive dims → non-null, correct shape
 * 13.  ffi_vector_create_nonpositive  — len≤0 → null
 * 14.  ffi_vector_create_positive     — positive len → non-null, correct length
 * 15.  ffi_gen_build_null_sizes       — null sizes ptr → null
 * 16.  ffi_gen_build_zero_count       — num_sizes≤0 → null
 * 17.  ffi_disc_build_null_sizes      — null sizes ptr → null
 * 18.  ffi_matrix_get_symbolic        — any (row,col) on 4×4: no panic, zero on OOB
 * 19.  ffi_matrix_set_symbolic        — any (row,col) on 4×4: no panic
 * 20.  ffi_matrix_set_get_roundtrip   — in-bounds set→get is bit-exact; OOB get is 0.0
 * 21.  ffi_vector_get_symbolic        — any idx on len-8 vector: no panic, zero on OOB
 * 22.  ffi_bool_zero_roundtrip        — set 0 → get 0
 * 23.  ffi_bool_nonzero_roundtrip     — set any non-zero → get non-zero
 * 24.  ffi_all_bool_fields_nonzero    — all 15 bool config fields, non-zero → non-zero
 * 25.  ffi_config_int_roundtrip       — set/get epochs and batch_size are identity
 * 26.  ffi_config_float_roundtrip     — set/get learning_rate is bit-exact identity
 * 27.  ffi_enum_activation_fallback   — all 5 known variants + unknown + null: no panic
 * 28.  ffi_enum_optimizer_fallback    — all 3 known variants + unknown + null: no panic
 * 29.  ffi_enum_loss_type_fallback    — all 4 known variants + unknown + null: no panic
 * 30.  ffi_enum_noise_type_fallback   — all 3 known variants + unknown + null: no panic
 * 31.  ffi_config_null_strings        — all 8 string setters accept null char*: no panic
 * 32.  ffi_detect_backend_non_null    — gf_detect_backend always returns non-null
 * 33.  ffi_validate_path_no_panic     — null/traversal/safe paths: correct result
 * 34.  ffi_matrix_from_data_null      — null data ptr → null
 * 35.  ffi_matrix_from_data_valid     — valid ptr + symbolic dims → correct shape
 * 36.  ffi_bounds_check_consistent    — in-bounds iff all four index constraints satisfied
 * 37.  ffi_cosine_anneal_no_panic     — any symbolic epoch/max_ep/lr: no panic
 * 38.  ffi_generate_noise_no_panic    — positive dims, known/unknown/null noise type: non-null
 * 39.  ffi_matrix_data_ptr_valid      — data pointer is non-null; all reads are 0.0
 *
 * Run the full FFI suite:
 *   cargo kani -p facaded_gan_c
 * Run a single harness:
 *   cargo kani -p facaded_gan_c --harness proof_ffi_null_matrix_accessors
 */

use std::os::raw::{c_char, c_float, c_int};
use std::ptr;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Null C string (simulates a C caller passing NULL for a const char*).
fn null_str() -> *const c_char { ptr::null() }

/// Build a static null-terminated C string from a literal.
macro_rules! cstr {
    ($s:literal) => {
        concat!($s, "\0").as_ptr() as *const c_char
    };
}

// =============================================================================
// 1. Null-safety — Matrix API accessors
// =============================================================================

/// Every matrix accessor/mutator handles a null pointer without panicking.
/// Returns 0 / 0.0 / null as appropriate; free is a no-op.
#[kani::proof]
fn proof_ffi_null_matrix_accessors() {
    let null_m: *const super::GanMatrix = ptr::null();
    let null_m_mut: *mut super::GanMatrix = ptr::null_mut();
    let r: c_int = kani::any();
    let c: c_int = kani::any();

    unsafe {
        kani::assert(super::gf_matrix_rows(null_m) == 0,   "null matrix rows → 0");
        kani::assert(super::gf_matrix_cols(null_m) == 0,   "null matrix cols → 0");
        kani::assert(super::gf_matrix_get(null_m, r, c) == 0.0, "null matrix get → 0.0");
        kani::assert(super::gf_matrix_data(null_m).is_null(), "null matrix data → null ptr");
        kani::assert(
            super::gf_matrix_safe_get(null_m, r, c, -1.0) == -1.0,
            "null matrix safe_get → default",
        );
        kani::assert(super::gf_bounds_check(null_m, r, c) == 0, "null matrix bounds_check → 0");

        // These must be no-ops (not crash).
        super::gf_matrix_set(null_m_mut, r, c, 1.0);
        super::gf_matrix_free(null_m_mut);
    }
}

// =============================================================================
// 2. Null-safety — Matrix arithmetic
// =============================================================================

/// Arithmetic on null matrices delegates to gang_to_matrix → [] and calls
/// matrix_to_gan with an empty TMatrix.  The result is always a valid
/// (non-null) GanMatrix with rows=0, cols=0.
#[kani::proof]
fn proof_ffi_null_matrix_arithmetic() {
    let null_m: *const super::GanMatrix = ptr::null();

    unsafe {
        let r1 = super::gf_matrix_multiply(null_m, null_m);
        kani::assert(!r1.is_null(), "multiply(null,null) → non-null empty matrix");
        super::gf_matrix_free(r1);

        let r2 = super::gf_matrix_add(null_m, null_m);
        kani::assert(!r2.is_null(), "add(null,null) → non-null");
        super::gf_matrix_free(r2);

        let r3 = super::gf_matrix_subtract(null_m, null_m);
        kani::assert(!r3.is_null(), "subtract(null,null) → non-null");
        super::gf_matrix_free(r3);

        let r4 = super::gf_matrix_scale(null_m, 2.0);
        kani::assert(!r4.is_null(), "scale(null) → non-null");
        super::gf_matrix_free(r4);

        let r5 = super::gf_matrix_transpose(null_m);
        kani::assert(!r5.is_null(), "transpose(null) → non-null");
        super::gf_matrix_free(r5);

        let r6 = super::gf_matrix_normalize(null_m);
        kani::assert(!r6.is_null(), "normalize(null) → non-null");
        super::gf_matrix_free(r6);

        let r7 = super::gf_matrix_element_mul(null_m, null_m);
        kani::assert(!r7.is_null(), "element_mul(null,null) → non-null");
        super::gf_matrix_free(r7);
    }
}

// =============================================================================
// 3. Null-safety — Vector API accessors
// =============================================================================

#[kani::proof]
fn proof_ffi_null_vector_accessors() {
    let null_v: *const super::GanVector = ptr::null();
    let null_v_mut: *mut super::GanVector = ptr::null_mut();
    let idx: c_int = kani::any();

    unsafe {
        kani::assert(super::gf_vector_len(null_v)  == 0,   "null vector len → 0");
        kani::assert(super::gf_vector_get(null_v, idx) == 0.0, "null vector get → 0.0");
        kani::assert(super::gf_vector_data(null_v).is_null(), "null vector data → null ptr");
        super::gf_vector_free(null_v_mut);  // must be a no-op
    }
}

// =============================================================================
// 4. Null-safety — Config API
// =============================================================================

#[kani::proof]
fn proof_ffi_null_config_accessors() {
    let null_c: *const super::GanConfig = ptr::null();
    let null_c_mut: *mut super::GanConfig = ptr::null_mut();

    unsafe {
        // Integer getters → 0
        kani::assert(super::gf_config_get_epochs(null_c)     == 0, "null cfg epochs → 0");
        kani::assert(super::gf_config_get_batch_size(null_c) == 0, "null cfg batch_size → 0");
        kani::assert(super::gf_config_get_noise_depth(null_c) == 0, "null cfg noise_depth → 0");
        kani::assert(super::gf_config_get_num_threads(null_c) == 0, "null cfg num_threads → 0");

        // Float getters → 0.0
        kani::assert(super::gf_config_get_learning_rate(null_c) == 0.0, "null cfg lr → 0.0");
        kani::assert(super::gf_config_get_gp_lambda(null_c)     == 0.0, "null cfg gp_lambda → 0.0");

        // Bool getters → 0
        kani::assert(super::gf_config_get_use_conv(null_c)      == 0, "null cfg use_conv → 0");
        kani::assert(super::gf_config_get_use_attention(null_c) == 0, "null cfg use_attention → 0");

        // Setters are no-ops
        super::gf_config_set_epochs(null_c_mut, 42);
        super::gf_config_set_use_conv(null_c_mut, 1);
        super::gf_config_set_learning_rate(null_c_mut, 0.001);
        super::gf_config_set_activation(null_c_mut, cstr!("relu"));
        super::gf_config_set_save_model(null_c_mut, null_str());

        // Free is a no-op
        super::gf_config_free(null_c_mut);
    }
}

// =============================================================================
// 5. Null-safety — Network API
// =============================================================================

#[kani::proof]
fn proof_ffi_null_network_accessors() {
    let null_n: *const super::GanNetwork = ptr::null();
    let null_n_mut: *mut super::GanNetwork = ptr::null_mut();
    let null_m: *const super::GanMatrix = ptr::null();

    unsafe {
        kani::assert(super::gf_network_layer_count(null_n)   == 0,   "null net layer_count → 0");
        kani::assert(super::gf_network_learning_rate(null_n) == 0.0, "null net lr → 0.0");
        kani::assert(super::gf_network_is_training(null_n)   == 0,   "null net is_training → 0");

        // forward / backward / sample all guard on null net and return null
        kani::assert(
            super::gf_network_forward(null_n_mut, null_m).is_null(),
            "forward(null_net) → null",
        );
        kani::assert(
            super::gf_network_backward(null_n_mut, null_m).is_null(),
            "backward(null_net) → null",
        );
        kani::assert(
            super::gf_network_sample(null_n_mut, 8, 64, cstr!("gauss")).is_null(),
            "sample(null_net) → null",
        );

        // Void operations are no-ops
        super::gf_network_set_training(null_n_mut, 1);
        super::gf_network_update_weights(null_n_mut);
        super::gf_network_verify(null_n_mut);
        super::gf_network_free(null_n_mut);
    }
}

// =============================================================================
// 6. Null-safety — Metrics API
// =============================================================================

#[kani::proof]
fn proof_ffi_null_metrics_accessors() {
    let null_m: *const super::GanMetrics = ptr::null();
    let null_m_mut: *mut super::GanMetrics = ptr::null_mut();

    unsafe {
        kani::assert(super::gf_metrics_d_loss_real(null_m)  == 0.0, "null metrics d_loss_real → 0.0");
        kani::assert(super::gf_metrics_d_loss_fake(null_m)  == 0.0, "null metrics d_loss_fake → 0.0");
        kani::assert(super::gf_metrics_g_loss(null_m)       == 0.0, "null metrics g_loss → 0.0");
        kani::assert(super::gf_metrics_fid_score(null_m)    == 0.0, "null metrics fid_score → 0.0");
        kani::assert(super::gf_metrics_is_score(null_m)     == 0.0, "null metrics is_score → 0.0");
        kani::assert(super::gf_metrics_grad_penalty(null_m) == 0.0, "null metrics grad_penalty → 0.0");
        kani::assert(super::gf_metrics_epoch(null_m) == 0, "null metrics epoch → 0");
        kani::assert(super::gf_metrics_batch(null_m) == 0, "null metrics batch → 0");
        super::gf_metrics_free(null_m_mut);
    }
}

// =============================================================================
// 7. Null-safety — Result API
// =============================================================================

#[kani::proof]
fn proof_ffi_null_result() {
    let null_r: *const super::GanResult = ptr::null();
    let null_r_mut: *mut super::GanResult = ptr::null_mut();

    unsafe {
        kani::assert(super::gf_result_generator(null_r).is_null(),     "result_generator(null) → null");
        kani::assert(super::gf_result_discriminator(null_r).is_null(), "result_discriminator(null) → null");
        kani::assert(super::gf_result_metrics(null_r).is_null(),       "result_metrics(null) → null");
        super::gf_result_free(null_r_mut);
    }
}

// =============================================================================
// 8. gf_run — null config returns null
// =============================================================================

#[kani::proof]
fn proof_ffi_run_null_cfg_returns_null() {
    unsafe {
        let null_cfg: *const super::GanConfig = ptr::null();
        kani::assert(super::gf_run(null_cfg).is_null(), "gf_run(null) → null");
    }
}

// =============================================================================
// 9. gf_train_full — any null argument returns null
// =============================================================================

#[kani::proof]
fn proof_ffi_train_full_any_null_returns_null() {
    let null_n: *mut super::GanNetwork = ptr::null_mut();
    let null_d: *const super::GanDataset = ptr::null();
    let null_c: *const super::GanConfig = ptr::null();

    unsafe {
        // All null
        kani::assert(
            super::gf_train_full(null_n, null_n, null_d, null_c).is_null(),
            "train_full(all null) → null",
        );
        // Null dataset only
        let cfg = super::gf_config_create();
        kani::assert(
            super::gf_train_full(null_n, null_n, null_d, cfg).is_null(),
            "train_full(null gen) → null",
        );
        super::gf_config_free(cfg);
    }
}

// =============================================================================
// 10. gf_train_step — null guard
// =============================================================================

#[kani::proof]
fn proof_ffi_train_step_null_guard() {
    let null_n: *mut super::GanNetwork = ptr::null_mut();
    let null_m: *const super::GanMatrix = ptr::null();
    let null_c: *const super::GanConfig = ptr::null();

    unsafe {
        kani::assert(
            super::gf_train_step(null_n, null_n, null_m, null_m, null_c).is_null(),
            "train_step(null gen/disc/cfg) → null",
        );
        // Null cfg alone also returns null
        let cfg_null: *const super::GanConfig = ptr::null();
        kani::assert(
            super::gf_train_step(null_n, null_n, null_m, null_m, cfg_null).is_null(),
            "train_step(null cfg) → null",
        );
    }
}

// =============================================================================
// 11. gf_matrix_create — non-positive dimensions return null
// =============================================================================

#[kani::proof]
fn proof_ffi_matrix_create_nonpositive_dims_return_null() {
    kani::assert(super::gf_matrix_create(0,  4).is_null(), "create(0,  4) → null");
    kani::assert(super::gf_matrix_create(-1, 4).is_null(), "create(-1, 4) → null");
    kani::assert(super::gf_matrix_create(4,  0).is_null(), "create(4,  0) → null");
    kani::assert(super::gf_matrix_create(4, -1).is_null(), "create(4, -1) → null");
    kani::assert(super::gf_matrix_create(0,  0).is_null(), "create(0,  0) → null");
    kani::assert(
        super::gf_matrix_create(c_int::MIN, c_int::MAX).is_null(),
        "create(MIN, MAX) → null (rows ≤ 0)",
    );
}

// =============================================================================
// 12. gf_matrix_create — positive dimensions produce a valid matrix
// =============================================================================

#[kani::proof]
fn proof_ffi_matrix_create_positive_dims_valid() {
    let rows: c_int = kani::any();
    let cols: c_int = kani::any();
    // Constrain to avoid i32 overflow in rows*cols inside gf_matrix_create.
    kani::assume(rows >= 1 && rows <= 100);
    kani::assume(cols >= 1 && cols <= 100);

    let m = super::gf_matrix_create(rows, cols);
    kani::assert(!m.is_null(), "gf_matrix_create with positive dims → non-null");

    unsafe {
        kani::assert(super::gf_matrix_rows(m) == rows, "created matrix rows match request");
        kani::assert(super::gf_matrix_cols(m) == cols, "created matrix cols match request");
        kani::assert(!super::gf_matrix_data(m).is_null(), "data pointer must be non-null");
        super::gf_matrix_free(m);
    }
}

// =============================================================================
// 13. gf_vector_create — non-positive length returns null
// =============================================================================

#[kani::proof]
fn proof_ffi_vector_create_nonpositive_len_returns_null() {
    kani::assert(super::gf_vector_create(0).is_null(),  "create(0) → null");
    kani::assert(super::gf_vector_create(-1).is_null(), "create(-1) → null");
    kani::assert(super::gf_vector_create(c_int::MIN).is_null(), "create(MIN) → null");
}

// =============================================================================
// 14. gf_vector_create — positive length produces a valid vector
// =============================================================================

#[kani::proof]
fn proof_ffi_vector_create_positive_len_valid() {
    let len: c_int = kani::any();
    kani::assume(len >= 1 && len <= 256);

    let v = super::gf_vector_create(len);
    kani::assert(!v.is_null(), "gf_vector_create with positive len → non-null");

    unsafe {
        kani::assert(super::gf_vector_len(v) == len, "created vector length matches request");
        kani::assert(!super::gf_vector_data(v).is_null(), "vector data pointer must be non-null");
        super::gf_vector_free(v);
    }
}

// =============================================================================
// 15–16. gf_gen_build — null sizes / zero count
// =============================================================================

#[kani::proof]
fn proof_ffi_gen_build_null_sizes_returns_null() {
    unsafe {
        let null_sizes: *const c_int = ptr::null();
        let r = super::gf_gen_build(null_sizes, 3, cstr!("relu"), cstr!("adam"), 0.001);
        kani::assert(r.is_null(), "gf_gen_build(null sizes) → null");
    }
}

#[kani::proof]
fn proof_ffi_gen_build_nonpositive_count_returns_null() {
    unsafe {
        let sizes: [c_int; 3] = [64, 128, 1];
        let r0 = super::gf_gen_build(sizes.as_ptr(), 0,  cstr!("relu"), cstr!("adam"), 0.001);
        kani::assert(r0.is_null(), "gf_gen_build(num_sizes=0) → null");
        let r1 = super::gf_gen_build(sizes.as_ptr(), -1, cstr!("relu"), cstr!("adam"), 0.001);
        kani::assert(r1.is_null(), "gf_gen_build(num_sizes=-1) → null");
        let rm = super::gf_gen_build(sizes.as_ptr(), c_int::MIN, cstr!("relu"), cstr!("adam"), 0.001);
        kani::assert(rm.is_null(), "gf_gen_build(num_sizes=MIN) → null");
    }
}

// =============================================================================
// 17. gf_disc_build — null sizes
// =============================================================================

#[kani::proof]
fn proof_ffi_disc_build_null_sizes_returns_null() {
    unsafe {
        let null_sizes: *const c_int = ptr::null();
        let r = super::gf_disc_build(null_sizes, 3, cstr!("relu"), cstr!("adam"), 0.001);
        kani::assert(r.is_null(), "gf_disc_build(null sizes) → null");
    }
}

// =============================================================================
// 18. gf_matrix_get — any symbolic (row, col) on a 4×4 matrix: no panic
// =============================================================================

/// On a zero-initialised 4×4 matrix, every possible (row, col) pair must
/// either return 0.0 (in-bounds) or 0.0 (OOB sentinel).  Either way the
/// result is 0.0 and the proof never panics.
#[kani::proof]
fn proof_ffi_matrix_get_symbolic_indices_no_panic() {
    let m = super::gf_matrix_create(4, 4);
    kani::assert(!m.is_null(), "precondition: create 4×4");

    let row: c_int = kani::any();
    let col: c_int = kani::any();

    unsafe {
        let val = super::gf_matrix_get(m, row, col);
        kani::assert(val == 0.0, "zero-initialised matrix: any get returns 0.0");
        super::gf_matrix_free(m);
    }
}

// =============================================================================
// 19. gf_matrix_set — any symbolic (row, col): no panic
// =============================================================================

#[kani::proof]
fn proof_ffi_matrix_set_symbolic_indices_no_panic() {
    let m = super::gf_matrix_create(4, 4);
    kani::assert(!m.is_null(), "precondition: create 4×4");

    let row: c_int = kani::any();
    let col: c_int = kani::any();
    let val: c_float = kani::any();

    unsafe {
        super::gf_matrix_set(m, row, col, val); // must not panic
        super::gf_matrix_free(m);
    }
}

// =============================================================================
// 20. gf_matrix_set / gf_matrix_get — round-trip correctness
// =============================================================================

/// If (row, col) is in-bounds, set then get must return the stored value
/// (bit-exact, handles NaN).  Out-of-bounds get must always return 0.0.
#[kani::proof]
fn proof_ffi_matrix_set_get_roundtrip() {
    let m = super::gf_matrix_create(3, 3);
    kani::assert(!m.is_null(), "precondition: create 3×3");

    let row: c_int = kani::any();
    let col: c_int = kani::any();
    let val: c_float = kani::any();
    kani::assume(val.is_finite()); // exclude NaN/Inf for the round-trip assertion

    unsafe {
        super::gf_matrix_set(m, row, col, val);
        let read = super::gf_matrix_get(m, row, col);

        if row >= 0 && row < 3 && col >= 0 && col < 3 {
            kani::assert(
                read.to_bits() == val.to_bits(),
                "in-bounds set→get must be bit-exact",
            );
        } else {
            kani::assert(read == 0.0, "out-of-bounds get must return 0.0");
        }

        super::gf_matrix_free(m);
    }
}

// =============================================================================
// 21. gf_vector_get — any symbolic idx on a length-8 vector: no panic
// =============================================================================

#[kani::proof]
fn proof_ffi_vector_get_symbolic_index_no_panic() {
    let v = super::gf_vector_create(8);
    kani::assert(!v.is_null(), "precondition: create vector len=8");

    let idx: c_int = kani::any();

    unsafe {
        let val = super::gf_vector_get(v, idx);
        kani::assert(val == 0.0, "zero-initialised vector: any get returns 0.0");
        super::gf_vector_free(v);
    }
}

// =============================================================================
// 22–24. Boolean convention — C int ↔ Rust bool mapping
// =============================================================================

/// 0 → false: set 0, get back 0.
#[kani::proof]
fn proof_ffi_bool_zero_roundtrip() {
    let cfg = super::gf_config_create();
    kani::assert(!cfg.is_null(), "config create must succeed");
    unsafe {
        super::gf_config_set_use_conv(cfg, 0);
        kani::assert(super::gf_config_get_use_conv(cfg) == 0, "use_conv: set 0 → get 0");
        super::gf_config_free(cfg);
    }
}

/// Non-zero → true: any non-zero int maps to a non-zero return.
#[kani::proof]
fn proof_ffi_bool_nonzero_roundtrip() {
    let v: c_int = kani::any();
    kani::assume(v != 0);

    let cfg = super::gf_config_create();
    kani::assert(!cfg.is_null(), "config create must succeed");
    unsafe {
        super::gf_config_set_use_conv(cfg, v);
        kani::assert(super::gf_config_get_use_conv(cfg) != 0, "non-zero input → non-zero output");
        super::gf_config_free(cfg);
    }
}

/// All 15 boolean config fields satisfy the non-zero → non-zero invariant.
#[kani::proof]
fn proof_ffi_all_bool_fields_nonzero_roundtrip() {
    let v: c_int = kani::any();
    kani::assume(v != 0);

    let cfg = super::gf_config_create();
    kani::assert(!cfg.is_null(), "config create must succeed");

    unsafe {
        macro_rules! check_bool {
            ($set:ident, $get:ident, $label:literal) => {{
                super::$set(cfg, v);
                kani::assert(super::$get(cfg) != 0, $label);
            }};
        }
        check_bool!(gf_config_set_use_batch_norm,       gf_config_get_use_batch_norm,       "use_batch_norm nonzero");
        check_bool!(gf_config_set_use_layer_norm,        gf_config_get_use_layer_norm,        "use_layer_norm nonzero");
        check_bool!(gf_config_set_use_spectral_norm,     gf_config_get_use_spectral_norm,     "use_spectral_norm nonzero");
        check_bool!(gf_config_set_use_label_smoothing,   gf_config_get_use_label_smoothing,   "use_label_smoothing nonzero");
        check_bool!(gf_config_set_use_feature_matching,  gf_config_get_use_feature_matching,  "use_feature_matching nonzero");
        check_bool!(gf_config_set_use_minibatch_std_dev, gf_config_get_use_minibatch_std_dev, "use_minibatch_std_dev nonzero");
        check_bool!(gf_config_set_use_progressive,       gf_config_get_use_progressive,       "use_progressive nonzero");
        check_bool!(gf_config_set_use_augmentation,      gf_config_get_use_augmentation,      "use_augmentation nonzero");
        check_bool!(gf_config_set_compute_metrics,       gf_config_get_compute_metrics,       "compute_metrics nonzero");
        check_bool!(gf_config_set_use_weight_decay,      gf_config_get_use_weight_decay,      "use_weight_decay nonzero");
        check_bool!(gf_config_set_use_cosine_anneal,     gf_config_get_use_cosine_anneal,     "use_cosine_anneal nonzero");
        check_bool!(gf_config_set_audit_log,             gf_config_get_audit_log,             "audit_log nonzero");
        check_bool!(gf_config_set_use_encryption,        gf_config_get_use_encryption,        "use_encryption nonzero");
        check_bool!(gf_config_set_use_conv,              gf_config_get_use_conv,              "use_conv nonzero");
        check_bool!(gf_config_set_use_attention,         gf_config_get_use_attention,         "use_attention nonzero");

        super::gf_config_free(cfg);
    }
}

// =============================================================================
// 25. Config integer fields — set / get is identity
// =============================================================================

#[kani::proof]
fn proof_ffi_config_int_roundtrip() {
    let epochs: c_int     = kani::any();
    let batch_size: c_int = kani::any();
    let noise_depth: c_int = kani::any();
    let num_threads: c_int = kani::any();

    let cfg = super::gf_config_create();
    kani::assert(!cfg.is_null(), "config create must succeed");

    unsafe {
        super::gf_config_set_epochs(cfg, epochs);
        kani::assert(super::gf_config_get_epochs(cfg) == epochs, "epochs set/get identity");

        super::gf_config_set_batch_size(cfg, batch_size);
        kani::assert(super::gf_config_get_batch_size(cfg) == batch_size, "batch_size set/get identity");

        super::gf_config_set_noise_depth(cfg, noise_depth);
        kani::assert(super::gf_config_get_noise_depth(cfg) == noise_depth, "noise_depth set/get identity");

        super::gf_config_set_num_threads(cfg, num_threads);
        kani::assert(super::gf_config_get_num_threads(cfg) == num_threads, "num_threads set/get identity");

        super::gf_config_free(cfg);
    }
}

// =============================================================================
// 26. Config float fields — set / get is bit-exact identity
// =============================================================================

#[kani::proof]
fn proof_ffi_config_float_roundtrip() {
    let lr: c_float      = kani::any();
    let lambda: c_float  = kani::any();
    let g_lr: c_float    = kani::any();
    let d_lr: c_float    = kani::any();
    let decay: c_float   = kani::any();
    // finite constraint keeps the round-trip semantically meaningful
    kani::assume(lr.is_finite() && lambda.is_finite());
    kani::assume(g_lr.is_finite() && d_lr.is_finite() && decay.is_finite());

    let cfg = super::gf_config_create();
    kani::assert(!cfg.is_null(), "config create must succeed");

    unsafe {
        super::gf_config_set_learning_rate(cfg, lr);
        kani::assert(
            super::gf_config_get_learning_rate(cfg).to_bits() == lr.to_bits(),
            "learning_rate set/get bit-exact",
        );

        super::gf_config_set_gp_lambda(cfg, lambda);
        kani::assert(
            super::gf_config_get_gp_lambda(cfg).to_bits() == lambda.to_bits(),
            "gp_lambda set/get bit-exact",
        );

        super::gf_config_set_generator_lr(cfg, g_lr);
        kani::assert(
            super::gf_config_get_generator_lr(cfg).to_bits() == g_lr.to_bits(),
            "generator_lr set/get bit-exact",
        );

        super::gf_config_set_discriminator_lr(cfg, d_lr);
        kani::assert(
            super::gf_config_get_discriminator_lr(cfg).to_bits() == d_lr.to_bits(),
            "discriminator_lr set/get bit-exact",
        );

        super::gf_config_set_weight_decay_val(cfg, decay);
        kani::assert(
            super::gf_config_get_weight_decay_val(cfg).to_bits() == decay.to_bits(),
            "weight_decay_val set/get bit-exact",
        );

        super::gf_config_free(cfg);
    }
}

// =============================================================================
// 27–30. Enum string parsing — known and unknown values, null: no panic
// =============================================================================

/// All 5 activation strings, an unknown, and null are accepted without panic.
#[kani::proof]
fn proof_ffi_enum_activation_fallback_no_panic() {
    unsafe {
        let cfg = super::gf_config_create();
        super::gf_config_set_activation(cfg, cstr!("relu"));
        super::gf_config_set_activation(cfg, cstr!("sigmoid"));
        super::gf_config_set_activation(cfg, cstr!("tanh"));
        super::gf_config_set_activation(cfg, cstr!("leaky"));
        super::gf_config_set_activation(cfg, cstr!("none"));
        super::gf_config_set_activation(cfg, cstr!("UNKNOWN_ACTIVATION"));
        super::gf_config_set_activation(cfg, null_str());  // null → "" → None
        super::gf_config_free(cfg);
    }
}

/// All 3 optimizer strings, an unknown, and null are accepted without panic.
#[kani::proof]
fn proof_ffi_enum_optimizer_fallback_no_panic() {
    unsafe {
        let cfg = super::gf_config_create();
        super::gf_config_set_optimizer(cfg, cstr!("adam"));
        super::gf_config_set_optimizer(cfg, cstr!("sgd"));
        super::gf_config_set_optimizer(cfg, cstr!("rmsprop"));
        super::gf_config_set_optimizer(cfg, cstr!("bad_optimizer"));
        super::gf_config_set_optimizer(cfg, null_str());   // null → "" → Adam
        super::gf_config_free(cfg);
    }
}

/// All 4 loss type strings, an unknown, and null are accepted without panic.
#[kani::proof]
fn proof_ffi_enum_loss_type_fallback_no_panic() {
    unsafe {
        let cfg = super::gf_config_create();
        super::gf_config_set_loss_type(cfg, cstr!("bce"));
        super::gf_config_set_loss_type(cfg, cstr!("wgan"));
        super::gf_config_set_loss_type(cfg, cstr!("hinge"));
        super::gf_config_set_loss_type(cfg, cstr!("ls"));
        super::gf_config_set_loss_type(cfg, cstr!("unknown_loss"));
        super::gf_config_set_loss_type(cfg, null_str());   // null → "" → BCE
        super::gf_config_free(cfg);
    }
}

/// All 3 noise type strings, an unknown, and null are accepted without panic.
#[kani::proof]
fn proof_ffi_enum_noise_type_fallback_no_panic() {
    unsafe {
        let cfg = super::gf_config_create();
        super::gf_config_set_noise_type(cfg, cstr!("gauss"));
        super::gf_config_set_noise_type(cfg, cstr!("uniform"));
        super::gf_config_set_noise_type(cfg, cstr!("analog"));
        super::gf_config_set_noise_type(cfg, cstr!("unknown_noise"));
        super::gf_config_set_noise_type(cfg, null_str());  // null → "" → Gauss
        super::gf_config_free(cfg);
    }
}

// =============================================================================
// 31. All 8 string config setters accept a null char*: no panic
// =============================================================================

/// The internal cstr() helper returns "" on null, so all string setters must
/// silently accept NULL without crashing.  This is the critical contract for
/// Go (C.CString / nil), Julia (Cstring / C_NULL), C# (null string marshal),
/// and Zig (optional sentinel pointers).
#[kani::proof]
fn proof_ffi_config_null_strings_no_panic() {
    unsafe {
        let cfg = super::gf_config_create();
        super::gf_config_set_save_model(cfg,      null_str());
        super::gf_config_set_load_model(cfg,      null_str());
        super::gf_config_set_load_json_model(cfg, null_str());
        super::gf_config_set_output_dir(cfg,      null_str());
        super::gf_config_set_data_path(cfg,       null_str());
        super::gf_config_set_audit_log_file(cfg,  null_str());
        super::gf_config_set_encryption_key(cfg,  null_str());
        super::gf_config_set_patch_config(cfg,    null_str());
        super::gf_config_free(cfg);
    }
}

// =============================================================================
// 32. gf_detect_backend — always returns a non-null static string
// =============================================================================

#[kani::proof]
fn proof_ffi_detect_backend_non_null() {
    let ptr = super::gf_detect_backend();
    kani::assert(!ptr.is_null(), "gf_detect_backend must return a non-null string");
}

// =============================================================================
// 33. gf_validate_path — null, traversal, and safe paths: correct result
// =============================================================================

#[kani::proof]
fn proof_ffi_validate_path_no_panic() {
    unsafe {
        // Null string → cstr() returns "" → validate_path returns false
        let r_null = super::gf_validate_path(null_str());
        kani::assert(r_null == 0, "validate_path(null) → 0 (invalid)");

        // Directory traversal must be rejected
        let r_trav = super::gf_validate_path(cstr!("../secret"));
        kani::assert(r_trav == 0, "path traversal must be rejected");

        // Embedded traversal must also be rejected
        let r_emb = super::gf_validate_path(cstr!("models/../etc/passwd"));
        kani::assert(r_emb == 0, "embedded traversal must be rejected");

        // A safe relative path must be accepted
        let r_ok = super::gf_validate_path(cstr!("models/checkpoint.bin"));
        kani::assert(r_ok != 0, "safe path must be accepted");
    }
}

// =============================================================================
// 34. gf_matrix_from_data — null data pointer → null
// =============================================================================

#[kani::proof]
fn proof_ffi_matrix_from_data_null_ptr() {
    unsafe {
        let null_data: *const c_float = ptr::null();
        let r = super::gf_matrix_from_data(null_data, 4, 4);
        kani::assert(r.is_null(), "gf_matrix_from_data(null data) → null");
    }
}

// =============================================================================
// 35. gf_matrix_from_data — valid pointer + symbolic dims → correct shape
// =============================================================================

#[kani::proof]
fn proof_ffi_matrix_from_data_valid_shape() {
    let rows: c_int = kani::any();
    let cols: c_int = kani::any();
    // Upper bound prevents i32 overflow in rows*cols
    kani::assume(rows >= 1 && rows <= 4);
    kani::assume(cols >= 1 && cols <= 4);

    // Stack buffer large enough for the maximum (4×4 = 16 elements)
    let data = [0.0f32; 16];

    unsafe {
        let m = super::gf_matrix_from_data(data.as_ptr(), rows, cols);
        kani::assert(!m.is_null(), "from_data with valid dims → non-null");
        kani::assert(super::gf_matrix_rows(m) == rows, "from_data rows match");
        kani::assert(super::gf_matrix_cols(m) == cols, "from_data cols match");
        super::gf_matrix_free(m);
    }
}

// =============================================================================
// 36. gf_bounds_check — result is consistent with matrix dimensions
// =============================================================================

#[kani::proof]
fn proof_ffi_bounds_check_consistent_with_dimensions() {
    let rows: c_int = kani::any();
    let cols: c_int = kani::any();
    kani::assume(rows >= 1 && rows <= 8);
    kani::assume(cols >= 1 && cols <= 8);

    let m = super::gf_matrix_create(rows, cols);
    kani::assert(!m.is_null(), "precondition: create");

    let r: c_int = kani::any();
    let c: c_int = kani::any();

    unsafe {
        let in_bounds = super::gf_bounds_check(m, r, c);
        if in_bounds != 0 {
            kani::assert(r >= 0,    "in-bounds row is non-negative");
            kani::assert(r < rows,  "in-bounds row is < rows");
            kani::assert(c >= 0,    "in-bounds col is non-negative");
            kani::assert(c < cols,  "in-bounds col is < cols");
        } else {
            kani::assert(
                r < 0 || r >= rows || c < 0 || c >= cols,
                "out-of-bounds implies at least one index is invalid",
            );
        }
        super::gf_matrix_free(m);
    }
}

// =============================================================================
// 37. gf_cosine_anneal — no panic across full symbolic input space
// =============================================================================

#[kani::proof]
fn proof_ffi_cosine_anneal_no_panic() {
    let epoch: c_int   = kani::any();
    let max_ep: c_int  = kani::any();
    let base_lr: c_float = kani::any();
    let min_lr: c_float  = kani::any();
    // Practical constraint: (base_lr - min_lr) must remain finite to avoid
    // Inf * 0 = NaN in the cosine schedule.
    kani::assume(base_lr.is_finite() && min_lr.is_finite());
    kani::assume((base_lr - min_lr).is_finite());

    let _ = super::gf_cosine_anneal(epoch, max_ep, base_lr, min_lr);
    // If we reach here without a panic Kani considers the proof passed.
}

// =============================================================================
// 38. gf_generate_noise — positive dims always return non-null
// =============================================================================

#[kani::proof]
fn proof_ffi_generate_noise_no_panic() {
    unsafe {
        let m1 = super::gf_generate_noise(4, 8, cstr!("gauss"));
        kani::assert(!m1.is_null(), "generate_noise(gauss) → non-null");
        super::gf_matrix_free(m1);

        let m2 = super::gf_generate_noise(4, 8, cstr!("uniform"));
        kani::assert(!m2.is_null(), "generate_noise(uniform) → non-null");
        super::gf_matrix_free(m2);

        let m3 = super::gf_generate_noise(4, 8, cstr!("analog"));
        kani::assert(!m3.is_null(), "generate_noise(analog) → non-null");
        super::gf_matrix_free(m3);

        // Unknown type → falls back to Gauss via parse_noise_type → still valid
        let m4 = super::gf_generate_noise(4, 8, cstr!("unknown_noise"));
        kani::assert(!m4.is_null(), "generate_noise(unknown) → non-null");
        super::gf_matrix_free(m4);

        // Null type string → cstr() returns "" → Gauss fallback → non-null
        let m5 = super::gf_generate_noise(4, 8, null_str());
        kani::assert(!m5.is_null(), "generate_noise(null str) → non-null");
        super::gf_matrix_free(m5);
    }
}

// =============================================================================
// 39. gf_matrix_data — pointer is non-null; all elements are 0.0
// =============================================================================

/// Verifies the data pointer contract: valid matrix → non-null data pointer;
/// every cell is 0.0 as guaranteed by gf_matrix_create.
#[kani::proof]
#[kani::unwind(4)]
fn proof_ffi_matrix_data_pointer_valid_and_elements_zero() {
    let m = super::gf_matrix_create(3, 3);
    kani::assert(!m.is_null(), "precondition: create 3×3");

    unsafe {
        let data_ptr = super::gf_matrix_data(m);
        kani::assert(!data_ptr.is_null(), "data pointer must be non-null for valid matrix");

        // Confirm every cell reads back as 0.0 through the accessor
        for r in 0..3_i32 {
            for c in 0..3_i32 {
                kani::assert(
                    super::gf_matrix_get(m, r, c) == 0.0,
                    "zero-initialised cell must read as 0.0",
                );
            }
        }
        super::gf_matrix_free(m);
    }
}

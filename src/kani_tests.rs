/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani verification entry point for facaded_gan_cuda.
 *
 * All proof harnesses live in the top-level kani/ directory, organized by
 * CISA "Secure by Design" requirement.  This file brings them into the
 * crate's module tree under the `cfg(kani)` guard so that `cargo kani`
 * discovers every #[kani::proof] function.
 *
 * Run the full suite:
 *   cargo kani --tests
 *
 * Run a single harness (e.g.):
 *   cargo kani --harness proof_safe_get_any_index_no_panic
 *
 * The #[path] attribute resolves relative to this source file (src/), so
 * "../kani/mod.rs" correctly points to the project-root kani/ directory.
 * Rust will then resolve submodule declarations inside mod.rs relative to
 * the kani/ directory itself.
 */

#[path = "../kani/mod.rs"]
pub mod kani;

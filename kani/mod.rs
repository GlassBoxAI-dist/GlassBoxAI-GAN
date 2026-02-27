/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification Test Suite — CISA "Secure by Design" Hardening
 *
 * 15 requirements mapped directly to #[kani::proof] harnesses:
 *   1.  bound_checks       — OOB-free indexing
 *   2.  pointer_validity   — Valid slice/reference access
 *   3.  no_panic           — No panic!, unwrap(), or expect()
 *   4.  integer_overflow   — Safe arithmetic (add/mul/sub)
 *   5.  div_by_zero        — Non-zero denominators proven
 *   6.  global_state       — Network/layer state invariants
 *   7.  deadlock_free      — Sequential locking order (no Mutex; proves absence)
 *   8.  input_sanitization — Bounded loops / recursion depth
 *   9.  result_coverage    — All Result/Option variants handled
 *  10.  memory_limits      — Allocation sizes stay finite & bounded
 *  11.  constant_time      — No secret-dependent branches
 *  12.  state_machine      — Training ↔ inference transitions guarded
 *  13.  enum_exhaustion    — All match arms covered, no hidden wildcard panics
 *  14.  float_sanity       — NaN/Inf never bypass logic checks
 *  15.  resource_limits    — Allocations respect a symbolic security budget
 *
 * Run the full suite:
 *   cargo kani --tests
 * Run a single harness:
 *   cargo kani --harness <harness_name>
 */

pub mod bound_checks;
pub mod pointer_validity;
pub mod no_panic;
pub mod integer_overflow;
pub mod div_by_zero;
pub mod global_state;
pub mod deadlock_free;
pub mod input_sanitization;
pub mod result_coverage;
pub mod memory_limits;
pub mod constant_time;
pub mod state_machine;
pub mod enum_exhaustion;
pub mod float_sanity;
pub mod resource_limits;

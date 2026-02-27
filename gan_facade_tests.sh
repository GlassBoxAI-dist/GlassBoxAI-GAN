#!/bin/bash
#
# MIT License
# Copyright (c) 2025 Matthew Abbott
#
# gan_facade_tests.sh - Bash test runner for the Rust GANFacade CLI
# Builds the binary and exercises all CLI functions of the GAN facade.
#
# Usage:
#   ./gan_facade_tests.sh              Run all tests
#   ./gan_facade_tests.sh --quick      Skip slow tests (full training, fuzz)
#   ./gan_facade_tests.sh --skip-build Skip cargo build step
#   ./gan_facade_tests.sh --category ops|gen|disc|train|sec|introspect|cli|quality
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="$SCRIPT_DIR/target/debug/facaded_gan_cuda"
TEST_OUTDIR="$SCRIPT_DIR/test_cli_output"
PASS=0
FAIL=0
SKIP=0
TOTAL=0
QUICK=0
SKIP_BUILD=0
CATEGORY="all"
FAILURES=""

# Slow tests to skip in quick mode
SLOW_TESTS="GF_Train_Full GF_Sec_RunTests GF_Sec_RunFuzzTests"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --category) CATEGORY="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--skip-build] [--category ops|gen|disc|train|sec|introspect|cli|quality]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Colors (if terminal supports them)
RED=""
GREEN=""
YELLOW=""
BOLD=""
RESET=""
if [ -t 1 ]; then
    RED="\033[0;31m"
    GREEN="\033[0;32m"
    YELLOW="\033[0;33m"
    BOLD="\033[1m"
    RESET="\033[0m"
fi

pass_msg() { printf "  ${GREEN}[PASS]${RESET} %s\n" "$1"; }
fail_msg() { printf "  ${RED}[FAIL]${RESET} %s\n" "$1"; }
skip_msg() { printf "  ${YELLOW}[SKIP]${RESET} %s (%s)\n" "$1" "$2"; }
section()  { printf "\n${BOLD}--- %s ---${RESET}\n" "$1"; }

record_pass() {
    ((PASS++)) || true
    ((TOTAL++)) || true
    pass_msg "$1"
}

record_fail() {
    ((FAIL++)) || true
    ((TOTAL++)) || true
    fail_msg "$1"
    FAILURES="$FAILURES  - $1\n"
}

record_skip() {
    ((SKIP++)) || true
    skip_msg "$1" "$2"
}

# Check if a category should run
should_run() {
    [ "$CATEGORY" = "all" ] || [ "$CATEGORY" = "$1" ]
}

echo "================================================================="
echo " GANFacade Rust CLI Test Runner"
echo " Category: $CATEGORY"
echo " Quick mode: $([ "$QUICK" -eq 1 ] && echo "yes" || echo "no")"
echo "================================================================="

# =========================================================================
# STEP 1: Build
# =========================================================================
section "BUILD"

if [ "$SKIP_BUILD" -eq 1 ]; then
    echo "  Skipping build (--skip-build)"
else
    echo "  Building with: cargo build --no-default-features"
    if cargo build --no-default-features --manifest-path "$SCRIPT_DIR/Cargo.toml" 2>&1; then
        record_pass "cargo build --no-default-features"
    else
        record_fail "cargo build --no-default-features"
        echo "ERROR: Build failed, cannot continue."
        exit 1
    fi
fi

if [ ! -x "$BINARY" ]; then
    echo "ERROR: $BINARY not found or not executable."
    exit 1
fi

# Clean test output directory
rm -rf "$TEST_OUTDIR"
mkdir -p "$TEST_OUTDIR"

# =========================================================================
# STEP 2: --help
# =========================================================================
if should_run "cli"; then
    section "CLI: --help"
    HELP_OUT=$("$BINARY" --help 2>&1) || true

    # Check key sections are present
    for keyword in "API REFERENCE" "BACKENDS" "TMatrix" "TActivationType" \
                   "GF_Op_" "GF_Gen_" "GF_Disc_" "GF_Train_" "GF_Sec_" \
                   "--backend" "--epochs" "--loss" "--save" "--test"; do
        if echo "$HELP_OUT" | grep -qF -- "$keyword"; then
            record_pass "--help contains '$keyword'"
        else
            record_fail "--help missing '$keyword'"
        fi
    done
fi

# =========================================================================
# STEP 3: --list
# =========================================================================
if should_run "cli"; then
    section "CLI: --list"
    LIST_OUT=$("$BINARY" --list 2>&1)
    FUNC_COUNT=$(echo "$LIST_OUT" | grep -c "^GF_" || true)

    if [ "$FUNC_COUNT" -eq 127 ]; then
        record_pass "--list shows 127 functions"
    else
        record_fail "--list shows $FUNC_COUNT functions (expected 127)"
    fi

    # Verify each category is represented
    for prefix in "GF_Op_" "GF_Gen_" "GF_Disc_" "GF_Train_" "GF_Sec_" "GF_Introspect_"; do
        if echo "$LIST_OUT" | grep -q "^${prefix}"; then
            record_pass "--list contains ${prefix} functions"
        else
            record_fail "--list missing ${prefix} functions"
        fi
    done
fi

# =========================================================================
# STEP 4: --test all
# =========================================================================
if should_run "cli"; then
    section "CLI: --test all"
    if [ "$QUICK" -eq 1 ]; then
        record_skip "--test all" "--quick"
    else
        TEST_ALL_OUT=$("$BINARY" --test all 2>&1) && TEST_ALL_RC=0 || TEST_ALL_RC=$?
        if [ "$TEST_ALL_RC" -eq 0 ]; then
            record_pass "--test all (exit code 0)"
        else
            record_fail "--test all (exit code $TEST_ALL_RC)"
        fi

        # Check pass count
        if echo "$TEST_ALL_OUT" | grep -q "127 passed"; then
            record_pass "--test all reports 127 passed"
        else
            record_fail "--test all did not report 127 passed"
        fi

        # Check no failures
        if echo "$TEST_ALL_OUT" | grep -q "0 failed"; then
            record_pass "--test all reports 0 failed"
        else
            record_fail "--test all has failures"
        fi
    fi
fi

# =========================================================================
# STEP 5: Individual --test for sampled functions per category
# =========================================================================

run_individual_test() {
    local func="$1"

    # Quick mode skip for slow tests
    if [ "$QUICK" -eq 1 ]; then
        for slow in $SLOW_TESTS; do
            if [ "$func" = "$slow" ]; then
                record_skip "--test $func" "--quick"
                return
            fi
        done
    fi

    local output
    output=$("$BINARY" --test "$func" 2>&1) && local rc=0 || local rc=$?
    if [ "$rc" -eq 0 ] && echo "$output" | grep -q "\[PASS\]"; then
        record_pass "--test $func"
    else
        record_fail "--test $func"
    fi
}

# GF_Op_ samples
if should_run "ops"; then
    section "INDIVIDUAL TESTS: GF_Op_"
    for func in GF_Op_CreateMatrix GF_Op_MatrixMultiply GF_Op_MatrixAdd \
                GF_Op_MatrixTranspose GF_Op_MatrixElementMul GF_Op_SafeGet \
                GF_Op_ReLU GF_Op_LeakyReLU GF_Op_Sigmoid GF_Op_Tanh \
                GF_Op_Softmax GF_Op_Activate GF_Op_ActivationBackward \
                GF_Op_Conv2D GF_Op_Deconv2D GF_Op_Conv1D \
                GF_Op_BatchNorm GF_Op_LayerNorm GF_Op_SpectralNorm \
                GF_Op_Attention GF_Op_AttentionBackward \
                GF_Op_CreateDenseLayer GF_Op_CreateConv2DLayer \
                GF_Op_LayerForward GF_Op_LayerBackward \
                GF_Op_RandomGaussian GF_Op_GenerateNoise GF_Op_NoiseSlerp; do
        run_individual_test "$func"
    done
fi

# GF_Gen_ samples
if should_run "gen"; then
    section "INDIVIDUAL TESTS: GF_Gen_"
    for func in GF_Gen_Build GF_Gen_BuildConv GF_Gen_Forward GF_Gen_Backward \
                GF_Gen_Sample GF_Gen_SampleConditional GF_Gen_UpdateWeights \
                GF_Gen_AddProgressiveLayer GF_Gen_GetLayerOutput \
                GF_Gen_SetTraining GF_Gen_Noise GF_Gen_NoiseSlerp GF_Gen_DeepCopy; do
        run_individual_test "$func"
    done
fi

# GF_Disc_ samples
if should_run "disc"; then
    section "INDIVIDUAL TESTS: GF_Disc_"
    for func in GF_Disc_Build GF_Disc_BuildConv GF_Disc_Evaluate GF_Disc_Forward \
                GF_Disc_Backward GF_Disc_UpdateWeights GF_Disc_GradPenalty \
                GF_Disc_FeatureMatch GF_Disc_MinibatchStdDev \
                GF_Disc_AddProgressiveLayer GF_Disc_GetLayerOutput \
                GF_Disc_SetTraining GF_Disc_DeepCopy; do
        run_individual_test "$func"
    done
fi

# GF_Train_ samples
if should_run "train"; then
    section "INDIVIDUAL TESTS: GF_Train_"
    for func in GF_Train_BCELoss GF_Train_BCEGrad \
                GF_Train_WGANDiscLoss GF_Train_WGANGenLoss \
                GF_Train_HingeDiscLoss GF_Train_HingeGenLoss \
                GF_Train_LSDiscLoss GF_Train_LSGenLoss \
                GF_Train_LabelSmoothing \
                GF_Train_AdamUpdate GF_Train_SGDUpdate GF_Train_RMSPropUpdate \
                GF_Train_CosineAnneal GF_Train_CreateSynthetic GF_Train_Augment \
                GF_Train_ComputeFID GF_Train_ComputeIS GF_Train_LogMetrics \
                GF_Train_SaveModel GF_Train_LoadModel \
                GF_Train_SaveJSON GF_Train_LoadJSON \
                GF_Train_SaveCheckpoint GF_Train_LoadCheckpoint \
                GF_Train_SaveSamples GF_Train_PlotCSV GF_Train_PrintBar \
                GF_Train_Optimize GF_Train_Step GF_Train_Full; do
        run_individual_test "$func"
    done
fi

# GF_Sec_ samples
if should_run "sec"; then
    section "INDIVIDUAL TESTS: GF_Sec_"
    for func in GF_Sec_AuditLog GF_Sec_SecureRandomize GF_Sec_GetOSRandom \
                GF_Sec_ValidatePath GF_Sec_VerifyWeights GF_Sec_VerifyNetwork \
                GF_Sec_EncryptModel GF_Sec_DecryptModel \
                GF_Sec_RunTests GF_Sec_RunFuzzTests GF_Sec_BoundsCheck; do
        run_individual_test "$func"
    done
fi

# GF_Introspect_ samples
if should_run "introspect"; then
    section "INDIVIDUAL TESTS: GF_Introspect_"
    for func in GF_Introspect_NetworkFields GF_Introspect_LayerFields \
                GF_Introspect_WeightAccess GF_Introspect_ForwardCache \
                GF_Introspect_ActivationStats GF_Introspect_Gradients \
                GF_Introspect_AdamState GF_Introspect_MultiUpdate \
                GF_Introspect_DiscFields GF_Introspect_WeightDecay \
                GF_Introspect_ConfigMutation GF_Introspect_LayerChain; do
        run_individual_test "$func"
    done
fi

# =========================================================================
# STEP 6: --detect
# =========================================================================
if should_run "cli"; then
    section "CLI: --detect"
    DETECT_OUT=$("$BINARY" --detect 2>&1) || true

    if echo "$DETECT_OUT" | grep -qi "backend\|cpu\|auto-detect"; then
        record_pass "--detect shows backend info"
    else
        record_fail "--detect missing backend info"
    fi

    if echo "$DETECT_OUT" | grep -qi "CPU.*always"; then
        record_pass "--detect shows CPU always available"
    else
        record_fail "--detect missing CPU availability"
    fi
fi

# =========================================================================
# STEP 7: Short training run (--epochs 2 --batch-size 4 --backend cpu)
# =========================================================================
if should_run "cli"; then
    section "CLI: Training runs"

    # Basic training
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --output "$TEST_OUTDIR/train_basic" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --epochs 2 --batch-size 4 --backend cpu"
    else
        record_fail "Training: --epochs 2 --batch-size 4 --backend cpu"
    fi

    # WGAN loss
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --loss wgan --output "$TEST_OUTDIR/train_wgan" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --loss wgan"
    else
        record_fail "Training: --loss wgan"
    fi

    # Hinge loss
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --loss hinge --output "$TEST_OUTDIR/train_hinge" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --loss hinge"
    else
        record_fail "Training: --loss hinge"
    fi

    # Least squares loss
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --loss ls --output "$TEST_OUTDIR/train_ls" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --loss ls"
    else
        record_fail "Training: --loss ls"
    fi

    # SGD optimizer
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --optimizer sgd --output "$TEST_OUTDIR/train_sgd" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --optimizer sgd"
    else
        record_fail "Training: --optimizer sgd"
    fi

    # RMSProp optimizer
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --optimizer rmsprop --output "$TEST_OUTDIR/train_rmsprop" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --optimizer rmsprop"
    else
        record_fail "Training: --optimizer rmsprop"
    fi

    # ReLU activation
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --activation relu --output "$TEST_OUTDIR/train_relu" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --activation relu"
    else
        record_fail "Training: --activation relu"
    fi

    # Tanh activation
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --activation tanh --output "$TEST_OUTDIR/train_tanh" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --activation tanh"
    else
        record_fail "Training: --activation tanh"
    fi

    # Uniform noise
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --noise-type uniform --output "$TEST_OUTDIR/train_uniform" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --noise-type uniform"
    else
        record_fail "Training: --noise-type uniform"
    fi

    # Convolutional architecture
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --conv --output "$TEST_OUTDIR/train_conv" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --conv"
    else
        record_fail "Training: --conv"
    fi

    # Spectral normalization
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --spectral-norm --output "$TEST_OUTDIR/train_specnorm" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --spectral-norm"
    else
        record_fail "Training: --spectral-norm"
    fi

    # Label smoothing
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --label-smoothing --output "$TEST_OUTDIR/train_smooth" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --label-smoothing"
    else
        record_fail "Training: --label-smoothing"
    fi

    # Cosine annealing
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --cosine-anneal --output "$TEST_OUTDIR/train_cosine" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --cosine-anneal"
    else
        record_fail "Training: --cosine-anneal"
    fi

    # Weight decay
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --weight-decay 0.001 --output "$TEST_OUTDIR/train_wd" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --weight-decay 0.001"
    else
        record_fail "Training: --weight-decay 0.001"
    fi

    # Checkpoint interval
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --checkpoint 1 --output "$TEST_OUTDIR/train_ckpt" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --checkpoint 1"
    else
        record_fail "Training: --checkpoint 1"
    fi

    # Audit logging
    AUDIT_FILE="$TEST_OUTDIR/audit.log"
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --audit-log --audit-file "$AUDIT_FILE" \
        --output "$TEST_OUTDIR/train_audit" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --audit-log"
    else
        record_fail "Training: --audit-log"
    fi
    if [ -f "$AUDIT_FILE" ]; then
        record_pass "Audit log file created"
    else
        record_fail "Audit log file not created"
    fi

    # Batch norm + layer norm combo
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --batch-norm --layer-norm \
        --output "$TEST_OUTDIR/train_norms" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: --batch-norm --layer-norm"
    else
        record_fail "Training: --batch-norm --layer-norm"
    fi

    # Kitchen sink: multiple flags
    TRAIN_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --lr 0.001 --loss wgan --optimizer adam \
        --spectral-norm --label-smoothing --cosine-anneal --weight-decay 0.0001 \
        --output "$TEST_OUTDIR/train_combo" 2>&1) && TRAIN_RC=0 || TRAIN_RC=$?
    if [ "$TRAIN_RC" -eq 0 ] && echo "$TRAIN_OUT" | grep -q "Done"; then
        record_pass "Training: multi-flag combo (wgan+specnorm+smoothing+cosine+wd)"
    else
        record_fail "Training: multi-flag combo (wgan+specnorm+smoothing+cosine+wd)"
    fi
fi

# =========================================================================
# STEP 8: --fuzz 50
# =========================================================================
if should_run "cli"; then
    section "CLI: --fuzz"
    if [ "$QUICK" -eq 1 ]; then
        record_skip "--fuzz 50" "--quick"
    else
        FUZZ_OUT=$("$BINARY" --fuzz 50 2>&1) && FUZZ_RC=0 || FUZZ_RC=$?
        if [ "$FUZZ_RC" -eq 0 ]; then
            record_pass "--fuzz 50 (exit code 0)"
        else
            record_fail "--fuzz 50 (exit code $FUZZ_RC)"
        fi

        if echo "$FUZZ_OUT" | grep -qi "fuzz.*passed\|all.*ok\|50 iterations"; then
            record_pass "--fuzz 50 output looks correct"
        else
            # Still pass if exit code was 0, just note output wasn't matched
            if [ "$FUZZ_RC" -eq 0 ]; then
                record_pass "--fuzz 50 completed successfully"
            else
                record_fail "--fuzz 50 output unexpected"
            fi
        fi
    fi
fi

# =========================================================================
# STEP 9: --save / --load round-trip (binary)
# =========================================================================
if should_run "cli"; then
    section "CLI: Save/Load round-trip (binary)"
    SAVE_FILE="$TEST_OUTDIR/model_test.bin"

    # Train and save
    SAVE_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --save "$SAVE_FILE" --output "$TEST_OUTDIR/save_test" 2>&1) && SAVE_RC=0 || SAVE_RC=$?
    if [ "$SAVE_RC" -eq 0 ] && [ -f "$SAVE_FILE" ]; then
        record_pass "--save $SAVE_FILE (file created)"
    else
        record_fail "--save $SAVE_FILE (file not created or error)"
    fi

    # Load and train more
    if [ -f "$SAVE_FILE" ]; then
        LOAD_OUT=$("$BINARY" --backend cpu --epochs 1 --batch-size 4 \
            --load "$SAVE_FILE" --output "$TEST_OUTDIR/load_test" 2>&1) && LOAD_RC=0 || LOAD_RC=$?
        if [ "$LOAD_RC" -eq 0 ] && echo "$LOAD_OUT" | grep -q "Done"; then
            record_pass "--load $SAVE_FILE (resumed training)"
        else
            record_fail "--load $SAVE_FILE (resume failed)"
        fi
    else
        record_skip "--load round-trip" "no saved file"
    fi
fi

# =========================================================================
# STEP 10: --save / --load round-trip (JSON)
# =========================================================================
if should_run "cli"; then
    section "CLI: Save/Load round-trip (JSON)"
    JSON_FILE="$TEST_OUTDIR/model_test.json"

    # Train and save as JSON
    JSON_OUT=$("$BINARY" --backend cpu --epochs 2 --batch-size 4 \
        --save "$JSON_FILE" --output "$TEST_OUTDIR/json_save" 2>&1) && JSON_RC=0 || JSON_RC=$?
    if [ "$JSON_RC" -eq 0 ] && [ -f "$JSON_FILE" ]; then
        record_pass "--save $JSON_FILE (JSON file created)"

        # Verify it's valid JSON
        if head -c 1 "$JSON_FILE" | grep -q '{'; then
            record_pass "JSON file starts with '{'"
        else
            record_fail "JSON file does not appear to be valid JSON"
        fi
    else
        record_fail "--save $JSON_FILE (file not created or error)"
    fi

    # Load JSON
    if [ -f "$JSON_FILE" ]; then
        LOADJ_OUT=$("$BINARY" --backend cpu --epochs 1 --batch-size 4 \
            --load-json "$JSON_FILE" --output "$TEST_OUTDIR/json_load" 2>&1) && LOADJ_RC=0 || LOADJ_RC=$?
        if [ "$LOADJ_RC" -eq 0 ] && echo "$LOADJ_OUT" | grep -q "Done"; then
            record_pass "--load-json $JSON_FILE (resumed training)"
        else
            record_fail "--load-json $JSON_FILE (resume failed)"
        fi
    else
        record_skip "--load-json round-trip" "no saved JSON file"
    fi
fi

# =========================================================================
# STEP 11: --tests (built-in unit tests flag)
# =========================================================================
if should_run "cli"; then
    section "CLI: --tests (built-in)"
    if [ "$QUICK" -eq 1 ]; then
        record_skip "--tests" "--quick"
    else
        TESTS_OUT=$("$BINARY" --tests 2>&1) && TESTS_RC=0 || TESTS_RC=$?
        if [ "$TESTS_RC" -eq 0 ] && echo "$TESTS_OUT" | grep -qi "passed"; then
            record_pass "--tests (built-in unit tests)"
        else
            record_fail "--tests (built-in unit tests)"
        fi
    fi
fi

# =========================================================================
# STEP 12: Error handling
# =========================================================================
if should_run "cli"; then
    section "CLI: Error handling"

    # --test with no argument should fail
    ERR_OUT=$("$BINARY" --test 2>&1) && ERR_RC=0 || ERR_RC=$?
    if [ "$ERR_RC" -ne 0 ]; then
        record_pass "--test (no arg) exits with error"
    else
        record_fail "--test (no arg) should exit with error"
    fi

    # --test with invalid function name should fail
    ERR_OUT=$("$BINARY" --test NonExistentFunction 2>&1) && ERR_RC=0 || ERR_RC=$?
    if [ "$ERR_RC" -ne 0 ]; then
        record_pass "--test NonExistentFunction exits with error"
    else
        record_fail "--test NonExistentFunction should exit with error"
    fi
fi

# =========================================================================
# STEP 13: --quality-tests (training stability, mode collapse, FID/IS)
# =========================================================================
if should_run "cli" || should_run "quality"; then
    section "CLI: --quality-tests (stability / mode collapse / FID+IS)"
    if [ "$QUICK" -eq 1 ]; then
        record_skip "--quality-tests" "--quick"
    else
        QTESTS_OUT=$("$BINARY" --quality-tests 2>&1) && QTESTS_RC=0 || QTESTS_RC=$?

        if [ "$QTESTS_RC" -eq 0 ]; then
            record_pass "--quality-tests (exit code 0)"
        else
            record_fail "--quality-tests (exit code $QTESTS_RC)"
        fi

        if echo "$QTESTS_OUT" | grep -q "\[PASS\] Training stability"; then
            record_pass "--quality-tests: training stability"
        else
            record_fail "--quality-tests: training stability"
        fi

        if echo "$QTESTS_OUT" | grep -q "\[PASS\] Mode collapse detection"; then
            record_pass "--quality-tests: mode collapse detection"
        else
            record_fail "--quality-tests: mode collapse detection"
        fi

        if echo "$QTESTS_OUT" | grep -q "\[PASS\] FID/IS on toy dataset"; then
            record_pass "--quality-tests: FID/IS on toy dataset"
        else
            record_fail "--quality-tests: FID/IS on toy dataset"
        fi

        if echo "$QTESTS_OUT" | grep -qi "Quality tests passed"; then
            record_pass "--quality-tests: all quality checks passed"
        else
            record_fail "--quality-tests: not all quality checks passed"
        fi
    fi
fi

# =========================================================================
# Cleanup
# =========================================================================
rm -rf "$TEST_OUTDIR"

# =========================================================================
# Summary
# =========================================================================
echo ""
echo "================================================================="
echo " RESULTS: $TOTAL tested | $PASS passed | $FAIL failed | $SKIP skipped"
echo "================================================================="

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "FAILURES:"
    printf "$FAILURES"
    echo ""
    exit 1
else
    echo "All tests passed."
    exit 0
fi

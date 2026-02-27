/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * GANFacade CLI — Rust/cudarc port with CPU/CUDA/OpenCL/Hybrid backends.
 */

use facaded_gan_cuda::backend::{self, ComputeBackend};
use facaded_gan_cuda::facade::*;
use facaded_gan_cuda::quality_tests;
use facaded_gan_cuda::tests;
use facaded_gan_cuda::types::*;
use std::env;

fn show_help() {
    println!("GANFacade - Facade Pattern Interface for GAN");
    println!("Version: 2.0  |  GAN Unit: v{}", GAN_VERSION);
    println!("MIT License (c) 2025 Matthew Abbott\n");
    println!("USAGE:");
    println!("  facaded_gan_cuda [options]");
    println!("  facaded_gan_cuda --test <function-name>");
    println!("  facaded_gan_cuda --test all\n");
    println!("OPTIONS:");
    println!("  --help, -h              Show this help and API reference");
    println!("  --test <function-name>  Test a specific GF_ function");
    println!("  --test all              Test all GF_ functions");
    println!("  --list                  List all testable function names");
    println!("  --backend <type>        Compute backend: cpu, cuda, opencl, hybrid, auto");
    println!("                          (default: auto-detect best available)");
    println!("  --detect                Show detected backend and exit");
    println!("  --epochs N              Training epochs (default 100)");
    println!("  --batch-size N          Batch size (default 32)");
    println!("  --lr F                  Learning rate (default 0.0002)");
    println!("  --gen-lr F              Generator learning rate (TTUR)");
    println!("  --disc-lr F             Discriminator learning rate (TTUR)");
    println!("  --noise-depth N         Noise vector depth (default 64)");
    println!("  --condition-size N      Conditional input size (default 0)");
    println!("  --optimizer <adam|sgd|rmsprop>  Optimizer (default adam)");
    println!("  --activation <relu|sigmoid|tanh|leaky|none>  Activation (default leaky)");
    println!("  --noise-type <gauss|uniform|analog>  Noise type (default gauss)");
    println!("  --loss <bce|wgan|hinge|ls>  Loss type (default bce)");
    println!("  --data-type <image|audio|vector>  Dataset type (default vector)");
    println!("  --conv                  Use convolutional architecture");
    println!("  --use-attention         Use attention layers");
    println!("  --batch-norm            Enable batch normalization");
    println!("  --layer-norm            Enable layer normalization");
    println!("  --spectral-norm         Enable spectral normalization");
    println!("  --progressive           Enable progressive growing");
    println!("  --max-res N             Max progressive resolution level");
    println!("  --label-smoothing       Enable label smoothing");
    println!("  --feature-matching      Enable feature matching loss");
    println!("  --minibatch-stddev      Enable minibatch std dev");
    println!("  --cosine-anneal         Enable cosine annealing LR schedule");
    println!("  --augment               Enable data augmentation");
    println!("  --weight-decay F        Weight decay value (enables weight decay)");
    println!("  --gp-lambda F           Gradient penalty lambda (default 10)");
    println!("  --metrics               Compute FID/IS metrics during training");
    println!("  --metric-interval N     Metric computation interval (epochs)");
    println!("  --checkpoint N          Checkpoint save interval (epochs)");
    println!("  --save <file>           Save model (.bin or .json)");
    println!("  --load <file>           Load pretrained model");
    println!("  --load-json <file>      Load pretrained model from JSON");
    println!("  --data <path>           Dataset path");
    println!("  --output <dir>          Output directory");
    println!("  --audit-log             Enable audit logging");
    println!("  --audit-file <path>     Audit log file path");
    println!("  --encrypt <key>         Encrypt saved model with key");
    println!("  --gbit N                Generator bit depth");
    println!("  --dbit N                Discriminator bit depth");
    println!("  --patch-config <str>    Patch configuration string");
    println!("  --tests                 Run built-in unit tests");
    println!("  --fuzz N                Run N fuzz test iterations");
    println!("  --quality-tests         Run training stability, mode collapse, FID/IS tests\n");
    println!("BACKENDS:");
    println!("  cpu      Pure Rust CPU (always available)");
    println!("  cuda     NVIDIA GPU via cudarc/nvrtc (requires CUDA toolkit)");
    println!("  opencl   GPU/CPU via OpenCL (requires OpenCL runtime)");
    println!("  hybrid   Auto-select best per operation");
    println!("  auto     Same as hybrid (default)\n");
    println!("=====================================================================");
    println!(" GAN FACADE API REFERENCE");
    println!("=====================================================================\n");
    println!("TYPES:");
    println!("  TMatrix = Vec<Vec<f32>>       TVector = Vec<f32>");
    println!("  TMatrixArray = Vec<TMatrix>    TKernelArray = Vec<TMatrix>");
    println!("  TLayer = struct   TNetwork = struct  TGANConfig = struct");
    println!("  TGANMetrics = struct   TDataset = struct\n");
    println!("ENUMS:");
    println!("  TActivationType = {{ReLU, Sigmoid, Tanh, LeakyReLU, None}}");
    println!("  TLayerType      = {{Dense, Conv2D, Deconv2D, Conv1D,");
    println!("                     BatchNorm, LayerNorm, SpectralNorm, Attention}}");
    println!("  TLossType       = {{BCE, WGANGP, Hinge, LeastSquares}}");
    println!("  TDataType       = {{Image, Audio, Vector}}");
    println!("  TNoiseType      = {{Gauss, Uniform, Analog}}");
    println!("  TOptimizer      = {{Adam, SGD, RMSProp}}\n");
    println!("--- GF_Op_ : LOW-LEVEL OPERATIONS ---");
    println!("  GF_Op_CreateMatrix(rows, cols) -> TMatrix");
    println!("  GF_Op_CreateVector(size) -> TVector");
    println!("  GF_Op_MatrixMultiply(A, B) -> TMatrix");
    println!("  GF_Op_MatrixAdd(A, B) -> TMatrix");
    println!("  GF_Op_MatrixSubtract(A, B) -> TMatrix");
    println!("  GF_Op_MatrixScale(A, s) -> TMatrix");
    println!("  GF_Op_MatrixTranspose(A) -> TMatrix");
    println!("  GF_Op_MatrixNormalize(A) -> TMatrix");
    println!("  GF_Op_MatrixElementMul(A, B) -> TMatrix");
    println!("  GF_Op_MatrixAddInPlace(A&, B)");
    println!("  GF_Op_MatrixScaleInPlace(A&, s)");
    println!("  GF_Op_MatrixClipInPlace(A&, lo, hi)");
    println!("  GF_Op_SafeGet(M, r, c, default) -> float");
    println!("  GF_Op_SafeSet(M&, r, c, value)");
    println!("  GF_Op_ReLU(A) -> TMatrix");
    println!("  GF_Op_LeakyReLU(A, alpha) -> TMatrix");
    println!("  GF_Op_Sigmoid(A) -> TMatrix");
    println!("  GF_Op_Tanh(A) -> TMatrix");
    println!("  GF_Op_Softmax(A) -> TMatrix");
    println!("  GF_Op_Activate(A, act) -> TMatrix");
    println!("  GF_Op_ActivationBackward(grad, pre, act) -> TMatrix");
    println!("  GF_Op_Conv2D(input, layer&) -> TMatrix");
    println!("  GF_Op_Conv2DBackward(layer&, grad) -> TMatrix");
    println!("  GF_Op_Deconv2D(input, layer&) -> TMatrix");
    println!("  GF_Op_Deconv2DBackward(layer&, grad) -> TMatrix");
    println!("  GF_Op_Conv1D(input, layer&) -> TMatrix");
    println!("  GF_Op_Conv1DBackward(layer&, grad) -> TMatrix");
    println!("  GF_Op_BatchNorm(input, layer&) -> TMatrix");
    println!("  GF_Op_BatchNormBackward(layer&, grad) -> TMatrix");
    println!("  GF_Op_LayerNorm(input, layer&) -> TMatrix");
    println!("  GF_Op_LayerNormBackward(layer&, grad) -> TMatrix");
    println!("  GF_Op_SpectralNorm(layer&) -> TMatrix");
    println!("  GF_Op_Attention(input, layer&) -> TMatrix");
    println!("  GF_Op_AttentionBackward(layer&, grad) -> TMatrix");
    println!("  GF_Op_CreateDenseLayer(in, out, act) -> TLayer");
    println!("  GF_Op_CreateConv2DLayer(iCh,oCh,k,s,p,w,h,act) -> TLayer");
    println!("  GF_Op_CreateDeconv2DLayer(iCh,oCh,k,s,p,w,h,act) -> TLayer");
    println!("  GF_Op_CreateConv1DLayer(iCh,oCh,k,s,p,len,act) -> TLayer");
    println!("  GF_Op_CreateBatchNormLayer(features) -> TLayer");
    println!("  GF_Op_CreateLayerNormLayer(features) -> TLayer");
    println!("  GF_Op_CreateAttentionLayer(dModel, nHeads) -> TLayer");
    println!("  GF_Op_LayerForward(layer&, input) -> TMatrix");
    println!("  GF_Op_LayerBackward(layer&, grad) -> TMatrix");
    println!("  GF_Op_InitLayerOptimizer(layer&, opt)");
    println!("  GF_Op_RandomGaussian -> float");
    println!("  GF_Op_RandomUniform(lo, hi) -> float");
    println!("  GF_Op_GenerateNoise(M&, size, depth, nt)");
    println!("  GF_Op_NoiseSlerp(v1, v2, t) -> TVector\n");
    println!("--- GF_Gen_ : GENERATOR ACTIONS ---");
    println!("  GF_Gen_Build(sizes, act, opt, lr) -> TNetwork");
    println!("  GF_Gen_BuildConv(noiseDim, condSz, baseCh, act, opt, lr) -> TNetwork");
    println!("  GF_Gen_Forward(gen&, input) -> TMatrix");
    println!("  GF_Gen_Backward(gen&, grad) -> TMatrix");
    println!("  GF_Gen_Sample(gen&, count, noiseDim, nt) -> TMatrix");
    println!("  GF_Gen_SampleConditional(gen&, count, noiseDim, condSz, nt, cond)");
    println!("  GF_Gen_UpdateWeights(gen&)");
    println!("  GF_Gen_AddProgressiveLayer(gen&, lvl)");
    println!("  GF_Gen_GetLayerOutput(gen&, idx) -> TMatrix");
    println!("  GF_Gen_SetTraining(gen&, bool)");
    println!("  GF_Gen_Noise(size, depth, nt) -> TMatrix");
    println!("  GF_Gen_NoiseSlerp(v1, v2, t) -> TVector");
    println!("  GF_Gen_DeepCopy(gen) -> TNetwork\n");
    println!("--- GF_Disc_ : DISCRIMINATOR ACTIONS ---");
    println!("  GF_Disc_Build(sizes, act, opt, lr) -> TNetwork");
    println!("  GF_Disc_BuildConv(iCh, iW, iH, condSz, baseCh, act, opt, lr)");
    println!("  GF_Disc_Evaluate(disc&, input) -> TMatrix");
    println!("  GF_Disc_Forward(disc&, input) -> TMatrix");
    println!("  GF_Disc_Backward(disc&, grad) -> TMatrix");
    println!("  GF_Disc_UpdateWeights(disc&)");
    println!("  GF_Disc_GradPenalty(disc&, real, fake, lambda) -> float");
    println!("  GF_Disc_FeatureMatch(disc&, real, fake, featLayer) -> float");
    println!("  GF_Disc_MinibatchStdDev(input) -> TMatrix");
    println!("  GF_Disc_AddProgressiveLayer(disc&, lvl)");
    println!("  GF_Disc_GetLayerOutput(disc&, idx) -> TMatrix");
    println!("  GF_Disc_SetTraining(disc&, bool)");
    println!("  GF_Disc_DeepCopy(disc) -> TNetwork\n");
    println!("--- GF_Train_ : TRAINING CONTROL ---");
    println!("  GF_Train_Full(gen&, disc&, ds&, cfg)");
    println!("  GF_Train_Step(gen&, disc&, batch, noise, cfg)");
    println!("  GF_Train_Optimize(net&)");
    println!("  GF_Train_AdamUpdate(p&, g, m&, v&, t, lr, b1, b2, e, wd)");
    println!("  GF_Train_SGDUpdate(p&, g, lr, wd)");
    println!("  GF_Train_RMSPropUpdate(p&, g, cache&, lr, decay, e, wd)");
    println!("  GF_Train_CosineAnneal(ep, maxEp, baseLR, minLR) -> float");
    println!("  GF_Train_BCELoss(pred, target) -> float");
    println!("  GF_Train_BCEGrad(pred, target) -> TMatrix");
    println!("  GF_Train_WGANDiscLoss(dReal, dFake) -> float");
    println!("  GF_Train_WGANGenLoss(dFake) -> float");
    println!("  GF_Train_HingeDiscLoss(dReal, dFake) -> float");
    println!("  GF_Train_HingeGenLoss(dFake) -> float");
    println!("  GF_Train_LSDiscLoss(dReal, dFake) -> float");
    println!("  GF_Train_LSGenLoss(dFake) -> float");
    println!("  GF_Train_LabelSmoothing(labels, lo, hi) -> TMatrix");
    println!("  GF_Train_LoadDataset(path, dt) -> TDataset");
    println!("  GF_Train_LoadBMP(path) -> TDataset");
    println!("  GF_Train_LoadWAV(path) -> TDataset");
    println!("  GF_Train_CreateSynthetic(count, features) -> TDataset");
    println!("  GF_Train_Augment(sample, dt) -> TMatrix");
    println!("  GF_Train_ComputeFID(realS, fakeS) -> float");
    println!("  GF_Train_ComputeIS(samples) -> float");
    println!("  GF_Train_LogMetrics(metrics, filename)");
    println!("  GF_Train_SaveModel(net, filename)");
    println!("  GF_Train_LoadModel(net&, filename)");
    println!("  GF_Train_SaveJSON(gen, disc, filename)");
    println!("  GF_Train_LoadJSON(gen&, disc&, filename)");
    println!("  GF_Train_SaveCheckpoint(gen, disc, ep, dir)");
    println!("  GF_Train_LoadCheckpoint(gen&, disc&, ep, dir)");
    println!("  GF_Train_SaveSamples(gen&, ep, dir, nDim, nt)");
    println!("  GF_Train_PlotCSV(fn, dLoss, gLoss, cnt)");
    println!("  GF_Train_PrintBar(dLoss, gLoss, width)\n");
    println!("--- GF_Sec_ : SECURITY & ENTROPY ---");
    println!("  GF_Sec_AuditLog(msg, logFile)        [NIST AU-2/AU-3]");
    println!("  GF_Sec_SecureRandomize                [/dev/urandom seed]");
    println!("  GF_Sec_GetOSRandom -> uint8_t         [/dev/urandom byte]");
    println!("  GF_Sec_ValidatePath(path) -> bool");
    println!("  GF_Sec_VerifyWeights(layer&)          [NaN/Inf clean]");
    println!("  GF_Sec_VerifyNetwork(net&)            [all layers]");
    println!("  GF_Sec_EncryptModel(in, out, key)     [NIST SC-28]");
    println!("  GF_Sec_DecryptModel(in, out, key)     [NIST SC-28]");
    println!("  GF_Sec_RunTests -> bool               [SA-11]");
    println!("  GF_Sec_RunFuzzTests(iterations) -> bool [SA-11]");
    println!("  GF_Sec_BoundsCheck(M, r, c) -> bool\n");
    println!("--- GF_LAYER : LAYER HANDLE (C-ABI) ---");
    println!("  GF_Layer_CreateDense(in, out, act)               -> GanLayer*");
    println!("  GF_Layer_CreateConv2D(iCh,oCh,k,s,p,w,h,act)    -> GanLayer*");
    println!("  GF_Layer_CreateDeconv2D(iCh,oCh,k,s,p,w,h,act)  -> GanLayer*");
    println!("  GF_Layer_CreateConv1D(iCh,oCh,k,s,p,len,act)    -> GanLayer*");
    println!("  GF_Layer_CreateBatchNorm(features)               -> GanLayer*");
    println!("  GF_Layer_CreateLayerNorm(features)               -> GanLayer*");
    println!("  GF_Layer_CreateAttention(dModel, nHeads)         -> GanLayer*");
    println!("  GF_Layer_Free(layer*)");
    println!("  GF_Layer_Forward(layer*, inp*)                   -> Matrix*");
    println!("  GF_Layer_Backward(layer*, grad*)                 -> Matrix*");
    println!("  GF_Layer_InitOptimizer(layer*, opt)");
    println!("  GF_Layer_Conv2D(inp*, layer*)                    -> Matrix*");
    println!("  GF_Layer_Conv2DBackward(layer*, grad*)           -> Matrix*");
    println!("  GF_Layer_Deconv2D(inp*, layer*)                  -> Matrix*");
    println!("  GF_Layer_Deconv2DBackward(layer*, grad*)         -> Matrix*");
    println!("  GF_Layer_Conv1D(inp*, layer*)                    -> Matrix*");
    println!("  GF_Layer_Conv1DBackward(layer*, grad*)           -> Matrix*");
    println!("  GF_Layer_BatchNorm(inp*, layer*)                 -> Matrix*");
    println!("  GF_Layer_BatchNormBackward(layer*, grad*)        -> Matrix*");
    println!("  GF_Layer_LayerNorm(inp*, layer*)                 -> Matrix*");
    println!("  GF_Layer_LayerNormBackward(layer*, grad*)        -> Matrix*");
    println!("  GF_Layer_SpectralNorm(layer*)                    -> Matrix*");
    println!("  GF_Layer_Attention(inp*, layer*)                 -> Matrix*");
    println!("  GF_Layer_AttentionBackward(layer*, grad*)        -> Matrix*");
    println!("  GF_Layer_VerifyWeights(layer*)\n");
    println!("--- GF_MATRIX_ARRAY : FID/IS CONTAINER ---");
    println!("  GF_MatrixArray_Create()                          -> GanMatrixArray*");
    println!("  GF_MatrixArray_Free(arr*)");
    println!("  GF_MatrixArray_Push(arr*, m*)");
    println!("  GF_MatrixArray_Len(arr*)                         -> int\n");
    println!("--- MATRIX EXTENSIONS ---");
    println!("  GF_Matrix_AddInPlace(a*, b*)");
    println!("  GF_Matrix_ScaleInPlace(a*, s)");
    println!("  GF_Matrix_ClipInPlace(a*, lo, hi)");
    println!("  GF_Matrix_SafeSet(m*, r, c, val)");
    println!("  GF_ActivationBackward(grad*, pre_act*, act)      -> Matrix*\n");
    println!("--- GENERATOR / DISCRIMINATOR EXTENSIONS ---");
    println!("  GF_Gen_SampleConditional(gen*, n, ndim, csz, nt, cond*) -> Matrix*");
    println!("  GF_Gen_AddProgressiveLayer(gen*, res_lvl)");
    println!("  GF_Gen_GetLayerOutput(gen*, idx)                 -> Matrix*");
    println!("  GF_Gen_DeepCopy(gen*)                            -> Network*");
    println!("  GF_Disc_Evaluate(disc*, inp*)                    -> Matrix*");
    println!("  GF_Disc_GradPenalty(disc*, real*, fake*, lambda) -> float");
    println!("  GF_Disc_FeatureMatch(disc*, real*, fake*, layer) -> float");
    println!("  GF_Disc_MinibatchStdDev(inp*)                    -> Matrix*");
    println!("  GF_Disc_AddProgressiveLayer(disc*, res_lvl)");
    println!("  GF_Disc_GetLayerOutput(disc*, idx)               -> Matrix*");
    println!("  GF_Disc_DeepCopy(disc*)                          -> Network*\n");
    println!("--- TRAINING EXTENSIONS ---");
    println!("  GF_Train_Optimize(net*)");
    println!("  GF_Train_AdamUpdate(p*,g*,m*,v*,t,lr,b1,b2,eps,wd)");
    println!("  GF_Train_SGDUpdate(p*,g*,lr,wd)");
    println!("  GF_Train_RMSPropUpdate(p*,g*,cache*,lr,decay,eps,wd)");
    println!("  GF_Train_LabelSmoothing(labels*, lo, hi)         -> Matrix*");
    println!("  GF_Train_LoadBMP(path)                           -> Dataset*");
    println!("  GF_Train_LoadWAV(path)                           -> Dataset*");
    println!("  GF_Train_Augment(sample*, data_type)             -> Matrix*");
    println!("  GF_Train_LogMetrics(m*, filename)");
    println!("  GF_Train_SaveSamples(gen*, ep, dir, ndim, nt)");
    println!("  GF_Train_PlotCSV(file, d_loss[], g_loss[], cnt)");
    println!("  GF_Train_PrintBar(d_loss, g_loss, width)");
    println!("  GF_Train_ComputeFID(real_arr*, fake_arr*)        -> float");
    println!("  GF_Train_ComputeIS(samples*)                     -> float\n");
    println!("EXAMPLES:");
    println!("  ./facaded_gan_cuda --help");
    println!("  ./facaded_gan_cuda --test GF_Op_MatrixMultiply");
    println!("  ./facaded_gan_cuda --test all");
    println!("  ./facaded_gan_cuda --list");
    println!("  ./facaded_gan_cuda --backend cuda --epochs 50 --lr 0.0002");
    println!("  ./facaded_gan_cuda --backend cpu --loss wgan");
    println!("  ./facaded_gan_cuda --detect");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Early exit commands: --help, --list, --test (before backend init)
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                show_help();
                return;
            }
            "--list" => {
                tests::list_functions();
                return;
            }
            "--test" => {
                gf_sec_secure_randomize();
                i += 1;
                if i >= args.len() {
                    eprintln!("ERROR: --test requires a function name or \"all\"");
                    std::process::exit(1);
                }
                let test_name = &args[i];
                if test_name == "all" {
                    println!("GANFacade --test all");
                    println!("GAN Unit v{}\n", GAN_VERSION);
                    if !tests::run_all_tests() {
                        std::process::exit(1);
                    }
                } else {
                    if tests::run_single_test(test_name) {
                        println!("[PASS] {}", test_name);
                    } else {
                        println!("[FAIL] {}", test_name);
                        std::process::exit(1);
                    }
                }
                return;
            }
            _ => {}
        }
        i += 1;
    }

    // Parse all arguments
    let mut backend_choice: Option<ComputeBackend> = None;
    let mut detect_only = false;
    let mut config = GANConfig::default();

    i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--detect" => { detect_only = true; }
            "--backend" => {
                i += 1;
                if i < args.len() {
                    backend_choice = Some(args[i].parse::<ComputeBackend>().unwrap_or_else(|e| {
                        eprintln!("Error: {}", e);
                        eprintln!("Valid options: cpu, cuda, opencl, hybrid, auto");
                        std::process::exit(1);
                    }));
                }
            }
            "--epochs" => {
                i += 1;
                if i < args.len() { config.epochs = args[i].parse().unwrap_or(100); }
            }
            "--batch-size" => {
                i += 1;
                if i < args.len() { config.batch_size = args[i].parse().unwrap_or(32); }
            }
            "--lr" => {
                i += 1;
                if i < args.len() { config.learning_rate = args[i].parse().unwrap_or(0.0002); }
            }
            "--gen-lr" => {
                i += 1;
                if i < args.len() { config.generator_lr = args[i].parse().unwrap_or(0.0); }
            }
            "--disc-lr" => {
                i += 1;
                if i < args.len() { config.discriminator_lr = args[i].parse().unwrap_or(0.0); }
            }
            "--noise-depth" => {
                i += 1;
                if i < args.len() { config.noise_depth = args[i].parse().unwrap_or(64); }
            }
            "--condition-size" => {
                i += 1;
                if i < args.len() { config.condition_size = args[i].parse().unwrap_or(0); }
            }
            "--gp-lambda" => {
                i += 1;
                if i < args.len() { config.gp_lambda = args[i].parse().unwrap_or(10.0); }
            }
            "--weight-decay" => {
                i += 1;
                if i < args.len() {
                    config.weight_decay_val = args[i].parse().unwrap_or(0.0001);
                    config.use_weight_decay = true;
                }
            }
            "--max-res" => {
                i += 1;
                if i < args.len() { config.max_res_level = args[i].parse().unwrap_or(4); }
            }
            "--metric-interval" => {
                i += 1;
                if i < args.len() { config.metric_interval = args[i].parse().unwrap_or(10); }
            }
            "--checkpoint" => {
                i += 1;
                if i < args.len() { config.checkpoint_interval = args[i].parse().unwrap_or(0); }
            }
            "--gbit" => {
                i += 1;
                if i < args.len() { config.generator_bits = args[i].parse().unwrap_or(16); }
            }
            "--dbit" => {
                i += 1;
                if i < args.len() { config.discriminator_bits = args[i].parse().unwrap_or(16); }
            }
            "--patch-config" => {
                i += 1;
                if i < args.len() { config.patch_config = args[i].clone(); }
            }
            "--optimizer" => {
                i += 1;
                if i < args.len() {
                    config.optimizer = match args[i].as_str() {
                        "adam" => Optimizer::Adam,
                        "sgd" => Optimizer::SGD,
                        "rmsprop" => Optimizer::RMSProp,
                        _ => Optimizer::Adam,
                    };
                }
            }
            "--activation" => {
                i += 1;
                if i < args.len() {
                    config.activation = match args[i].as_str() {
                        "relu" => ActivationType::ReLU,
                        "sigmoid" => ActivationType::Sigmoid,
                        "tanh" => ActivationType::Tanh,
                        "leaky" => ActivationType::LeakyReLU,
                        "none" => ActivationType::None,
                        _ => ActivationType::LeakyReLU,
                    };
                }
            }
            "--noise-type" => {
                i += 1;
                if i < args.len() {
                    config.noise_type = match args[i].as_str() {
                        "gauss" => NoiseType::Gauss,
                        "uniform" => NoiseType::Uniform,
                        "analog" => NoiseType::Analog,
                        _ => NoiseType::Gauss,
                    };
                }
            }
            "--loss" => {
                i += 1;
                if i < args.len() {
                    config.loss_type = match args[i].as_str() {
                        "bce" => LossType::BCE,
                        "wgan" | "wgan-gp" => LossType::WGANGP,
                        "hinge" => LossType::Hinge,
                        "ls" => LossType::LeastSquares,
                        _ => LossType::BCE,
                    };
                }
            }
            "--data-type" => {
                i += 1;
                if i < args.len() {
                    config.data_type = match args[i].as_str() {
                        "image" => DataType::Image,
                        "audio" => DataType::Audio,
                        _ => DataType::Vector,
                    };
                }
            }
            "--conv" => { config.use_conv = true; }
            "--use-attention" => { config.use_attention = true; }
            "--batch-norm" => { config.use_batch_norm = true; }
            "--layer-norm" => { config.use_layer_norm = true; }
            "--spectral-norm" => { config.use_spectral_norm = true; }
            "--progressive" => { config.use_progressive = true; }
            "--label-smoothing" => { config.use_label_smoothing = true; }
            "--feature-matching" => { config.use_feature_matching = true; }
            "--minibatch-stddev" => { config.use_minibatch_std_dev = true; }
            "--cosine-anneal" => { config.use_cosine_anneal = true; }
            "--augment" => { config.use_augmentation = true; }
            "--metrics" => { config.compute_metrics = true; }
            "--audit-log" => { config.audit_log = true; }
            "--audit-file" => {
                i += 1;
                if i < args.len() { config.audit_log_file = args[i].clone(); }
            }
            "--encrypt" => {
                i += 1;
                if i < args.len() {
                    config.use_encryption = true;
                    config.encryption_key = args[i].clone();
                }
            }
            "--save" => {
                i += 1;
                if i < args.len() { config.save_model = args[i].clone(); }
            }
            "--load" => {
                i += 1;
                if i < args.len() { config.load_model = args[i].clone(); }
            }
            "--load-json" => {
                i += 1;
                if i < args.len() { config.load_json_model = args[i].clone(); }
            }
            "--data" => {
                i += 1;
                if i < args.len() { config.data_path = args[i].clone(); }
            }
            "--output" => {
                i += 1;
                if i < args.len() { config.output_dir = args[i].clone(); }
            }
            "--tests" => { config.run_tests = true; }
            "--quality-tests" => { config.run_quality_tests = true; }
            "--fuzz" => {
                config.run_fuzz = true;
                i += 1;
                if i < args.len() {
                    config.fuzz_iterations = args[i].parse().unwrap_or(100);
                } else {
                    config.fuzz_iterations = 100;
                    i -= 1; // no value given, rewind
                }
            }
            _ => {}
        }
        i += 1;
    }

    // Initialize backend
    let chosen = backend_choice.unwrap_or_else(|| {
        let best = backend::detect_best_backend();
        eprintln!("[Backend] Auto-detected: {}", best);
        best
    });

    if detect_only {
        let best = backend::detect_best_backend();
        println!("Auto-detected backend: {}", best);
        println!("\nAvailability:");
        println!("  CPU:    always");
        #[cfg(feature = "cuda")]
        {
            let cuda_ok = cudarc::driver::CudaDevice::new(0).is_ok();
            println!("  CUDA:   {}", if cuda_ok { "available" } else { "not available" });
        }
        #[cfg(not(feature = "cuda"))]
        println!("  CUDA:   not compiled (enable 'cuda' feature)");
        #[cfg(feature = "opencl")]
        {
            let cl_ok = !ocl::Platform::list().is_empty();
            println!("  OpenCL: {}", if cl_ok { "available" } else { "not available" });
        }
        #[cfg(not(feature = "opencl"))]
        println!("  OpenCL: not compiled (enable 'opencl' feature)");
        return;
    }

    backend::init_backend(chosen);
    let be = backend::get_backend();

    // Handle --tests and --fuzz
    if config.run_tests {
        gf_sec_secure_randomize();
        if gf_sec_run_tests() {
            println!("Tests passed.");
        } else {
            println!("Tests had failures.");
        }
        return;
    }
    if config.run_fuzz {
        gf_sec_secure_randomize();
        gf_sec_run_fuzz_tests(config.fuzz_iterations);
        return;
    }
    if config.run_quality_tests {
        gf_sec_secure_randomize();
        if quality_tests::run_quality_tests() {
            println!("Quality tests passed.");
            std::process::exit(0);
        } else {
            println!("Quality tests had failures.");
            std::process::exit(1);
        }
    }

    // Audit
    if config.audit_log {
        gf_sec_audit_log(
            &format!("GAN started with config: epochs={}", config.epochs),
            &config.audit_log_file,
        );
    }

    println!("GAN Facade v{}", GAN_VERSION);
    println!("Backend: {}\n", be.name());

    println!("Configuration:");
    println!("  Epochs: {}", config.epochs);
    println!("  Batch Size: {}", config.batch_size);
    println!("  Noise Depth: {}", config.noise_depth);
    println!("  Learning Rate: {}", config.learning_rate);
    println!("  Loss: {:?}", config.loss_type);
    println!("  Conv: {}", config.use_conv);
    println!("  Attention: {}", config.use_attention);
    println!("  Condition Size: {}", config.condition_size);
    println!();

    let result = gf_run(&config);

    println!("Generator layers: {}", result.generator.layer_count);
    println!("\nDone.");
}

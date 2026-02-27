// MIT License  Copyright (c) 2025 Matthew Abbott
//
// Zig wrapper for the facaded_gan_c C library.
//
// PREREQUISITES
// -------------
// Build the native library first:
//   cargo build --release -p facaded_gan_c
//
// EXAMPLE
// -------
//   const fg = @import("facaded_gan");
//
//   fg.initBackend("cpu");
//   var cfg = try fg.Config.init();
//   defer cfg.deinit();
//   cfg.setEpochs(2);
//   cfg.setBatchSize(8);
//   var result = try fg.run(&cfg);
//   defer result.deinit();
//   var m = try result.metrics();
//   defer m.deinit();
//   std.debug.print("g_loss: {d:.4}\n", .{m.gLoss()});

const std = @import("std");

// ── Error set ────────────────────────────────────────────────────────────────

pub const Error = error{
    /// A gf_* function returned a null pointer.
    AllocationFailed,
};

// ── Opaque C handle types ─────────────────────────────────────────────────────

const RawMatrix  = opaque {};
const RawVector  = opaque {};
const RawNetwork = opaque {};
const RawDataset = opaque {};
const RawConfig  = opaque {};
const RawMetrics = opaque {};
const RawResult  = opaque {};

// ── C extern declarations ─────────────────────────────────────────────────────

// Matrix
extern fn gf_matrix_create(rows: c_int, cols: c_int) ?*RawMatrix;
extern fn gf_matrix_from_data(data: [*]const f32, rows: c_int, cols: c_int) ?*RawMatrix;
extern fn gf_matrix_free(m: *RawMatrix) void;
extern fn gf_matrix_data(m: *const RawMatrix) [*]const f32;
extern fn gf_matrix_rows(m: *const RawMatrix) c_int;
extern fn gf_matrix_cols(m: *const RawMatrix) c_int;
extern fn gf_matrix_get(m: *const RawMatrix, row: c_int, col: c_int) f32;
extern fn gf_matrix_set(m: *RawMatrix, row: c_int, col: c_int, val: f32) void;
extern fn gf_matrix_multiply(a: *const RawMatrix, b: *const RawMatrix) ?*RawMatrix;
extern fn gf_matrix_add(a: *const RawMatrix, b: *const RawMatrix) ?*RawMatrix;
extern fn gf_matrix_subtract(a: *const RawMatrix, b: *const RawMatrix) ?*RawMatrix;
extern fn gf_matrix_scale(a: *const RawMatrix, s: f32) ?*RawMatrix;
extern fn gf_matrix_transpose(a: *const RawMatrix) ?*RawMatrix;
extern fn gf_matrix_normalize(a: *const RawMatrix) ?*RawMatrix;
extern fn gf_matrix_element_mul(a: *const RawMatrix, b: *const RawMatrix) ?*RawMatrix;
extern fn gf_matrix_safe_get(m: *const RawMatrix, r: c_int, c: c_int, def: f32) f32;

// Vector
extern fn gf_vector_create(len: c_int) ?*RawVector;
extern fn gf_vector_free(v: *RawVector) void;
extern fn gf_vector_data(v: *const RawVector) [*]const f32;
extern fn gf_vector_len(v: *const RawVector) c_int;
extern fn gf_vector_get(v: *const RawVector, idx: c_int) f32;
extern fn gf_vector_noise_slerp(v1: *const RawVector, v2: *const RawVector, t: f32) ?*RawVector;

// Config
extern fn gf_config_create() ?*RawConfig;
extern fn gf_config_free(cfg: *RawConfig) void;
// int getters/setters
extern fn gf_config_get_epochs(c: *const RawConfig) c_int;
extern fn gf_config_set_epochs(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_batch_size(c: *const RawConfig) c_int;
extern fn gf_config_set_batch_size(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_noise_depth(c: *const RawConfig) c_int;
extern fn gf_config_set_noise_depth(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_condition_size(c: *const RawConfig) c_int;
extern fn gf_config_set_condition_size(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_generator_bits(c: *const RawConfig) c_int;
extern fn gf_config_set_generator_bits(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_discriminator_bits(c: *const RawConfig) c_int;
extern fn gf_config_set_discriminator_bits(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_max_res_level(c: *const RawConfig) c_int;
extern fn gf_config_set_max_res_level(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_metric_interval(c: *const RawConfig) c_int;
extern fn gf_config_set_metric_interval(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_checkpoint_interval(c: *const RawConfig) c_int;
extern fn gf_config_set_checkpoint_interval(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_fuzz_iterations(c: *const RawConfig) c_int;
extern fn gf_config_set_fuzz_iterations(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_num_threads(c: *const RawConfig) c_int;
extern fn gf_config_set_num_threads(c: *RawConfig, v: c_int) void;
// float getters/setters
extern fn gf_config_get_learning_rate(c: *const RawConfig) f32;
extern fn gf_config_set_learning_rate(c: *RawConfig, v: f32) void;
extern fn gf_config_get_gp_lambda(c: *const RawConfig) f32;
extern fn gf_config_set_gp_lambda(c: *RawConfig, v: f32) void;
extern fn gf_config_get_generator_lr(c: *const RawConfig) f32;
extern fn gf_config_set_generator_lr(c: *RawConfig, v: f32) void;
extern fn gf_config_get_discriminator_lr(c: *const RawConfig) f32;
extern fn gf_config_set_discriminator_lr(c: *RawConfig, v: f32) void;
extern fn gf_config_get_weight_decay_val(c: *const RawConfig) f32;
extern fn gf_config_set_weight_decay_val(c: *RawConfig, v: f32) void;
// bool getters/setters (C int: 0=false, !=0=true)
extern fn gf_config_get_use_batch_norm(c: *const RawConfig) c_int;
extern fn gf_config_set_use_batch_norm(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_use_layer_norm(c: *const RawConfig) c_int;
extern fn gf_config_set_use_layer_norm(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_use_spectral_norm(c: *const RawConfig) c_int;
extern fn gf_config_set_use_spectral_norm(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_use_label_smoothing(c: *const RawConfig) c_int;
extern fn gf_config_set_use_label_smoothing(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_use_feature_matching(c: *const RawConfig) c_int;
extern fn gf_config_set_use_feature_matching(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_use_minibatch_std_dev(c: *const RawConfig) c_int;
extern fn gf_config_set_use_minibatch_std_dev(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_use_progressive(c: *const RawConfig) c_int;
extern fn gf_config_set_use_progressive(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_use_augmentation(c: *const RawConfig) c_int;
extern fn gf_config_set_use_augmentation(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_compute_metrics(c: *const RawConfig) c_int;
extern fn gf_config_set_compute_metrics(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_use_weight_decay(c: *const RawConfig) c_int;
extern fn gf_config_set_use_weight_decay(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_use_cosine_anneal(c: *const RawConfig) c_int;
extern fn gf_config_set_use_cosine_anneal(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_audit_log(c: *const RawConfig) c_int;
extern fn gf_config_set_audit_log(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_use_encryption(c: *const RawConfig) c_int;
extern fn gf_config_set_use_encryption(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_use_conv(c: *const RawConfig) c_int;
extern fn gf_config_set_use_conv(c: *RawConfig, v: c_int) void;
extern fn gf_config_get_use_attention(c: *const RawConfig) c_int;
extern fn gf_config_set_use_attention(c: *RawConfig, v: c_int) void;
// string setters
extern fn gf_config_set_save_model(c: *RawConfig, v: [*:0]const u8) void;
extern fn gf_config_set_load_model(c: *RawConfig, v: [*:0]const u8) void;
extern fn gf_config_set_load_json_model(c: *RawConfig, v: [*:0]const u8) void;
extern fn gf_config_set_output_dir(c: *RawConfig, v: [*:0]const u8) void;
extern fn gf_config_set_data_path(c: *RawConfig, v: [*:0]const u8) void;
extern fn gf_config_set_audit_log_file(c: *RawConfig, v: [*:0]const u8) void;
extern fn gf_config_set_encryption_key(c: *RawConfig, v: [*:0]const u8) void;
extern fn gf_config_set_patch_config(c: *RawConfig, v: [*:0]const u8) void;
// enum setters (string-based)
extern fn gf_config_set_activation(c: *RawConfig, v: [*:0]const u8) void;
extern fn gf_config_set_noise_type(c: *RawConfig, v: [*:0]const u8) void;
extern fn gf_config_set_optimizer(c: *RawConfig, v: [*:0]const u8) void;
extern fn gf_config_set_loss_type(c: *RawConfig, v: [*:0]const u8) void;
extern fn gf_config_set_data_type(c: *RawConfig, v: [*:0]const u8) void;

// Network
extern fn gf_gen_build(sizes: [*]const c_int, num_sizes: c_int, act: [*:0]const u8, opt: [*:0]const u8, lr: f32) ?*RawNetwork;
extern fn gf_gen_build_conv(noise_dim: c_int, cond_sz: c_int, base_ch: c_int, act: [*:0]const u8, opt: [*:0]const u8, lr: f32) ?*RawNetwork;
extern fn gf_disc_build(sizes: [*]const c_int, num_sizes: c_int, act: [*:0]const u8, opt: [*:0]const u8, lr: f32) ?*RawNetwork;
extern fn gf_disc_build_conv(in_ch: c_int, in_w: c_int, in_h: c_int, cond_sz: c_int, base_ch: c_int, act: [*:0]const u8, opt: [*:0]const u8, lr: f32) ?*RawNetwork;
extern fn gf_network_free(net: *RawNetwork) void;
extern fn gf_network_layer_count(net: *const RawNetwork) c_int;
extern fn gf_network_learning_rate(net: *const RawNetwork) f32;
extern fn gf_network_is_training(net: *const RawNetwork) c_int;
extern fn gf_network_forward(net: *RawNetwork, inp: *const RawMatrix) ?*RawMatrix;
extern fn gf_network_backward(net: *RawNetwork, grad_out: *const RawMatrix) ?*RawMatrix;
extern fn gf_network_update_weights(net: *RawNetwork) void;
extern fn gf_network_set_training(net: *RawNetwork, training: c_int) void;
extern fn gf_network_sample(net: *RawNetwork, count: c_int, noise_dim: c_int, noise_type: [*:0]const u8) ?*RawMatrix;
extern fn gf_network_verify(net: *RawNetwork) void;
extern fn gf_network_save(net: *const RawNetwork, path: [*:0]const u8) void;
extern fn gf_network_load(net: *RawNetwork, path: [*:0]const u8) void;

// Dataset
extern fn gf_dataset_create_synthetic(count: c_int, features: c_int) ?*RawDataset;
extern fn gf_dataset_load(path: [*:0]const u8, data_type: [*:0]const u8) ?*RawDataset;
extern fn gf_dataset_free(ds: *RawDataset) void;
extern fn gf_dataset_count(ds: *const RawDataset) c_int;

// Training
extern fn gf_train_full(gen: *RawNetwork, disc: *RawNetwork, ds: *const RawDataset, cfg: *const RawConfig) ?*RawMetrics;
extern fn gf_train_step(gen: *RawNetwork, disc: *RawNetwork, real_batch: *const RawMatrix, noise: *const RawMatrix, cfg: *const RawConfig) ?*RawMetrics;
extern fn gf_train_save_json(gen: *const RawNetwork, disc: *const RawNetwork, path: [*:0]const u8) void;
extern fn gf_train_load_json(gen: *RawNetwork, disc: *RawNetwork, path: [*:0]const u8) void;
extern fn gf_train_save_checkpoint(gen: *const RawNetwork, disc: *const RawNetwork, ep: c_int, dir: [*:0]const u8) void;
extern fn gf_train_load_checkpoint(gen: *RawNetwork, disc: *RawNetwork, ep: c_int, dir: [*:0]const u8) void;

// Metrics
extern fn gf_metrics_free(m: *RawMetrics) void;
extern fn gf_metrics_d_loss_real(m: *const RawMetrics) f32;
extern fn gf_metrics_d_loss_fake(m: *const RawMetrics) f32;
extern fn gf_metrics_g_loss(m: *const RawMetrics) f32;
extern fn gf_metrics_fid_score(m: *const RawMetrics) f32;
extern fn gf_metrics_is_score(m: *const RawMetrics) f32;
extern fn gf_metrics_grad_penalty(m: *const RawMetrics) f32;
extern fn gf_metrics_epoch(m: *const RawMetrics) c_int;
extern fn gf_metrics_batch(m: *const RawMetrics) c_int;

// High-level run
extern fn gf_run(cfg: *const RawConfig) ?*RawResult;
extern fn gf_result_free(r: *RawResult) void;
extern fn gf_result_generator(r: *const RawResult) ?*RawNetwork;
extern fn gf_result_discriminator(r: *const RawResult) ?*RawNetwork;
extern fn gf_result_metrics(r: *const RawResult) ?*RawMetrics;

// Loss
extern fn gf_bce_loss(pred: *const RawMatrix, target: *const RawMatrix) f32;
extern fn gf_bce_grad(pred: *const RawMatrix, target: *const RawMatrix) ?*RawMatrix;
extern fn gf_wgan_disc_loss(d_real: *const RawMatrix, d_fake: *const RawMatrix) f32;
extern fn gf_wgan_gen_loss(d_fake: *const RawMatrix) f32;
extern fn gf_hinge_disc_loss(d_real: *const RawMatrix, d_fake: *const RawMatrix) f32;
extern fn gf_hinge_gen_loss(d_fake: *const RawMatrix) f32;
extern fn gf_ls_disc_loss(d_real: *const RawMatrix, d_fake: *const RawMatrix) f32;
extern fn gf_ls_gen_loss(d_fake: *const RawMatrix) f32;
extern fn gf_cosine_anneal(epoch: c_int, max_ep: c_int, base_lr: f32, min_lr: f32) f32;

// Activations
extern fn gf_activate(a: *const RawMatrix, act_type: [*:0]const u8) ?*RawMatrix;
extern fn gf_relu(a: *const RawMatrix) ?*RawMatrix;
extern fn gf_sigmoid(a: *const RawMatrix) ?*RawMatrix;
extern fn gf_tanh_m(a: *const RawMatrix) ?*RawMatrix;
extern fn gf_leaky_relu(a: *const RawMatrix, alpha: f32) ?*RawMatrix;
extern fn gf_softmax(a: *const RawMatrix) ?*RawMatrix;

// Random / Noise
extern fn gf_random_gaussian() f32;
extern fn gf_random_uniform(lo: f32, hi: f32) f32;
extern fn gf_generate_noise(size: c_int, depth: c_int, noise_type: [*:0]const u8) ?*RawMatrix;

// Security
extern fn gf_validate_path(path: [*:0]const u8) c_int;
extern fn gf_audit_log(msg: [*:0]const u8, log_file: [*:0]const u8) void;
extern fn gf_bounds_check(m: *const RawMatrix, r: c_int, c: c_int) c_int;

// Backend
extern fn gf_init_backend(name: [*:0]const u8) void;
extern fn gf_detect_backend() [*:0]const u8;
extern fn gf_secure_randomize() void;

// ── Internal helper ───────────────────────────────────────────────────────────

inline fn ib(v: bool) c_int { return @as(c_int, @intFromBool(v)); }

// ── Matrix ────────────────────────────────────────────────────────────────────

pub const Matrix = struct {
    ptr: *RawMatrix,

    /// Create a zero-filled nrows×ncols matrix.
    pub fn init(nrows: c_int, ncols: c_int) Error!Matrix {
        return Matrix{ .ptr = gf_matrix_create(nrows, ncols) orelse return error.AllocationFailed };
    }

    /// Wrap a caller-supplied flat row-major slice into a new matrix.
    pub fn fromSlice(slice: []const f32, nrows: c_int, ncols: c_int) Error!Matrix {
        return Matrix{ .ptr = gf_matrix_from_data(slice.ptr, nrows, ncols) orelse return error.AllocationFailed };
    }

    pub fn deinit(self: Matrix) void { gf_matrix_free(self.ptr); }

    pub fn rows(self: Matrix) c_int  { return gf_matrix_rows(self.ptr); }
    pub fn cols(self: Matrix) c_int  { return gf_matrix_cols(self.ptr); }

    pub fn get(self: Matrix, row: c_int, col: c_int) f32 {
        return gf_matrix_get(self.ptr, row, col);
    }
    pub fn set(self: Matrix, row: c_int, col: c_int, val: f32) void {
        gf_matrix_set(self.ptr, row, col, val);
    }
    pub fn safeGet(self: Matrix, r: c_int, c: c_int, def: f32) f32 {
        return gf_matrix_safe_get(self.ptr, r, c, def);
    }

    /// Slice view of the internal flat row-major data.  Valid until deinit().
    pub fn data(self: Matrix) []const f32 {
        const n = @as(usize, @intCast(self.rows() * self.cols()));
        return gf_matrix_data(self.ptr)[0..n];
    }

    pub fn multiply(self: Matrix, other: Matrix) Error!Matrix {
        return Matrix{ .ptr = gf_matrix_multiply(self.ptr, other.ptr) orelse return error.AllocationFailed };
    }
    pub fn add(self: Matrix, other: Matrix) Error!Matrix {
        return Matrix{ .ptr = gf_matrix_add(self.ptr, other.ptr) orelse return error.AllocationFailed };
    }
    pub fn subtract(self: Matrix, other: Matrix) Error!Matrix {
        return Matrix{ .ptr = gf_matrix_subtract(self.ptr, other.ptr) orelse return error.AllocationFailed };
    }
    pub fn scale(self: Matrix, s: f32) Error!Matrix {
        return Matrix{ .ptr = gf_matrix_scale(self.ptr, s) orelse return error.AllocationFailed };
    }
    pub fn transpose(self: Matrix) Error!Matrix {
        return Matrix{ .ptr = gf_matrix_transpose(self.ptr) orelse return error.AllocationFailed };
    }
    pub fn normalize(self: Matrix) Error!Matrix {
        return Matrix{ .ptr = gf_matrix_normalize(self.ptr) orelse return error.AllocationFailed };
    }
    pub fn elementMul(self: Matrix, other: Matrix) Error!Matrix {
        return Matrix{ .ptr = gf_matrix_element_mul(self.ptr, other.ptr) orelse return error.AllocationFailed };
    }

    pub fn boundsCheck(self: Matrix, r: c_int, c: c_int) bool {
        return gf_bounds_check(self.ptr, r, c) != 0;
    }

    // ── Activations ─────────────────────────────────────────────────────────

    /// act_type: "relu" | "sigmoid" | "tanh" | "leaky" | "none"
    pub fn activate(self: Matrix, act_type: [:0]const u8) Error!Matrix {
        return Matrix{ .ptr = gf_activate(self.ptr, act_type.ptr) orelse return error.AllocationFailed };
    }
    pub fn relu(self: Matrix) Error!Matrix {
        return Matrix{ .ptr = gf_relu(self.ptr) orelse return error.AllocationFailed };
    }
    pub fn sigmoid(self: Matrix) Error!Matrix {
        return Matrix{ .ptr = gf_sigmoid(self.ptr) orelse return error.AllocationFailed };
    }
    pub fn tanh(self: Matrix) Error!Matrix {
        return Matrix{ .ptr = gf_tanh_m(self.ptr) orelse return error.AllocationFailed };
    }
    pub fn leakyRelu(self: Matrix, alpha: f32) Error!Matrix {
        return Matrix{ .ptr = gf_leaky_relu(self.ptr, alpha) orelse return error.AllocationFailed };
    }
    pub fn softmax(self: Matrix) Error!Matrix {
        return Matrix{ .ptr = gf_softmax(self.ptr) orelse return error.AllocationFailed };
    }
};

// ── Vector ────────────────────────────────────────────────────────────────────

pub const Vector = struct {
    ptr: *RawVector,

    pub fn init(n: c_int) Error!Vector {
        return Vector{ .ptr = gf_vector_create(n) orelse return error.AllocationFailed };
    }

    pub fn deinit(self: Vector) void { gf_vector_free(self.ptr); }

    pub fn len(self: Vector) c_int  { return gf_vector_len(self.ptr); }
    pub fn get(self: Vector, idx: c_int) f32 { return gf_vector_get(self.ptr, idx); }

    /// Slice view of the internal data.  Valid until deinit().
    pub fn data(self: Vector) []const f32 {
        const n = @as(usize, @intCast(self.len()));
        return gf_vector_data(self.ptr)[0..n];
    }

    /// Spherical linear interpolation between two noise vectors. t ∈ [0, 1].
    pub fn noiseSlerp(self: Vector, other: Vector, t: f32) Error!Vector {
        return Vector{ .ptr = gf_vector_noise_slerp(self.ptr, other.ptr, t) orelse return error.AllocationFailed };
    }
};

// ── Config ────────────────────────────────────────────────────────────────────

pub const Config = struct {
    ptr: *RawConfig,

    /// Create a config with library defaults.
    pub fn init() Error!Config {
        return Config{ .ptr = gf_config_create() orelse return error.AllocationFailed };
    }

    pub fn deinit(self: Config) void { gf_config_free(self.ptr); }

    // ── Integer fields ───────────────────────────────────────────────────────
    pub fn epochs(self: Config) c_int              { return gf_config_get_epochs(self.ptr); }
    pub fn setEpochs(self: Config, v: c_int) void  { gf_config_set_epochs(self.ptr, v); }

    pub fn batchSize(self: Config) c_int                { return gf_config_get_batch_size(self.ptr); }
    pub fn setBatchSize(self: Config, v: c_int) void    { gf_config_set_batch_size(self.ptr, v); }

    pub fn noiseDepth(self: Config) c_int               { return gf_config_get_noise_depth(self.ptr); }
    pub fn setNoiseDepth(self: Config, v: c_int) void   { gf_config_set_noise_depth(self.ptr, v); }

    pub fn conditionSize(self: Config) c_int              { return gf_config_get_condition_size(self.ptr); }
    pub fn setConditionSize(self: Config, v: c_int) void  { gf_config_set_condition_size(self.ptr, v); }

    pub fn generatorBits(self: Config) c_int              { return gf_config_get_generator_bits(self.ptr); }
    pub fn setGeneratorBits(self: Config, v: c_int) void  { gf_config_set_generator_bits(self.ptr, v); }

    pub fn discriminatorBits(self: Config) c_int               { return gf_config_get_discriminator_bits(self.ptr); }
    pub fn setDiscriminatorBits(self: Config, v: c_int) void   { gf_config_set_discriminator_bits(self.ptr, v); }

    pub fn maxResLevel(self: Config) c_int              { return gf_config_get_max_res_level(self.ptr); }
    pub fn setMaxResLevel(self: Config, v: c_int) void  { gf_config_set_max_res_level(self.ptr, v); }

    pub fn metricInterval(self: Config) c_int               { return gf_config_get_metric_interval(self.ptr); }
    pub fn setMetricInterval(self: Config, v: c_int) void   { gf_config_set_metric_interval(self.ptr, v); }

    pub fn checkpointInterval(self: Config) c_int               { return gf_config_get_checkpoint_interval(self.ptr); }
    pub fn setCheckpointInterval(self: Config, v: c_int) void   { gf_config_set_checkpoint_interval(self.ptr, v); }

    pub fn fuzzIterations(self: Config) c_int               { return gf_config_get_fuzz_iterations(self.ptr); }
    pub fn setFuzzIterations(self: Config, v: c_int) void   { gf_config_set_fuzz_iterations(self.ptr, v); }

    pub fn numThreads(self: Config) c_int               { return gf_config_get_num_threads(self.ptr); }
    pub fn setNumThreads(self: Config, v: c_int) void   { gf_config_set_num_threads(self.ptr, v); }

    // ── Float fields ─────────────────────────────────────────────────────────
    pub fn learningRate(self: Config) f32              { return gf_config_get_learning_rate(self.ptr); }
    pub fn setLearningRate(self: Config, v: f32) void  { gf_config_set_learning_rate(self.ptr, v); }

    pub fn gpLambda(self: Config) f32              { return gf_config_get_gp_lambda(self.ptr); }
    pub fn setGpLambda(self: Config, v: f32) void  { gf_config_set_gp_lambda(self.ptr, v); }

    pub fn generatorLr(self: Config) f32              { return gf_config_get_generator_lr(self.ptr); }
    pub fn setGeneratorLr(self: Config, v: f32) void  { gf_config_set_generator_lr(self.ptr, v); }

    pub fn discriminatorLr(self: Config) f32              { return gf_config_get_discriminator_lr(self.ptr); }
    pub fn setDiscriminatorLr(self: Config, v: f32) void  { gf_config_set_discriminator_lr(self.ptr, v); }

    pub fn weightDecayVal(self: Config) f32              { return gf_config_get_weight_decay_val(self.ptr); }
    pub fn setWeightDecayVal(self: Config, v: f32) void  { gf_config_set_weight_decay_val(self.ptr, v); }

    // ── Bool fields ──────────────────────────────────────────────────────────
    pub fn useBatchNorm(self: Config) bool               { return gf_config_get_use_batch_norm(self.ptr) != 0; }
    pub fn setUseBatchNorm(self: Config, v: bool) void   { gf_config_set_use_batch_norm(self.ptr, ib(v)); }

    pub fn useLayerNorm(self: Config) bool               { return gf_config_get_use_layer_norm(self.ptr) != 0; }
    pub fn setUseLayerNorm(self: Config, v: bool) void   { gf_config_set_use_layer_norm(self.ptr, ib(v)); }

    pub fn useSpectralNorm(self: Config) bool             { return gf_config_get_use_spectral_norm(self.ptr) != 0; }
    pub fn setUseSpectralNorm(self: Config, v: bool) void { gf_config_set_use_spectral_norm(self.ptr, ib(v)); }

    pub fn useLabelSmoothing(self: Config) bool               { return gf_config_get_use_label_smoothing(self.ptr) != 0; }
    pub fn setUseLabelSmoothing(self: Config, v: bool) void   { gf_config_set_use_label_smoothing(self.ptr, ib(v)); }

    pub fn useFeatureMatching(self: Config) bool               { return gf_config_get_use_feature_matching(self.ptr) != 0; }
    pub fn setUseFeatureMatching(self: Config, v: bool) void   { gf_config_set_use_feature_matching(self.ptr, ib(v)); }

    pub fn useMinibatchStdDev(self: Config) bool               { return gf_config_get_use_minibatch_std_dev(self.ptr) != 0; }
    pub fn setUseMinibatchStdDev(self: Config, v: bool) void   { gf_config_set_use_minibatch_std_dev(self.ptr, ib(v)); }

    pub fn useProgressive(self: Config) bool               { return gf_config_get_use_progressive(self.ptr) != 0; }
    pub fn setUseProgressive(self: Config, v: bool) void   { gf_config_set_use_progressive(self.ptr, ib(v)); }

    pub fn useAugmentation(self: Config) bool               { return gf_config_get_use_augmentation(self.ptr) != 0; }
    pub fn setUseAugmentation(self: Config, v: bool) void   { gf_config_set_use_augmentation(self.ptr, ib(v)); }

    pub fn computeMetrics(self: Config) bool               { return gf_config_get_compute_metrics(self.ptr) != 0; }
    pub fn setComputeMetrics(self: Config, v: bool) void   { gf_config_set_compute_metrics(self.ptr, ib(v)); }

    pub fn useWeightDecay(self: Config) bool               { return gf_config_get_use_weight_decay(self.ptr) != 0; }
    pub fn setUseWeightDecay(self: Config, v: bool) void   { gf_config_set_use_weight_decay(self.ptr, ib(v)); }

    pub fn useCosineAnneal(self: Config) bool               { return gf_config_get_use_cosine_anneal(self.ptr) != 0; }
    pub fn setUseCosineAnneal(self: Config, v: bool) void   { gf_config_set_use_cosine_anneal(self.ptr, ib(v)); }

    pub fn auditLog(self: Config) bool               { return gf_config_get_audit_log(self.ptr) != 0; }
    pub fn setAuditLog(self: Config, v: bool) void   { gf_config_set_audit_log(self.ptr, ib(v)); }

    pub fn useEncryption(self: Config) bool               { return gf_config_get_use_encryption(self.ptr) != 0; }
    pub fn setUseEncryption(self: Config, v: bool) void   { gf_config_set_use_encryption(self.ptr, ib(v)); }

    pub fn useConv(self: Config) bool               { return gf_config_get_use_conv(self.ptr) != 0; }
    pub fn setUseConv(self: Config, v: bool) void   { gf_config_set_use_conv(self.ptr, ib(v)); }

    pub fn useAttention(self: Config) bool               { return gf_config_get_use_attention(self.ptr) != 0; }
    pub fn setUseAttention(self: Config, v: bool) void   { gf_config_set_use_attention(self.ptr, ib(v)); }

    // ── String / enum setters ────────────────────────────────────────────────
    pub fn setSaveModel(self: Config, v: [:0]const u8) void       { gf_config_set_save_model(self.ptr, v.ptr); }
    pub fn setLoadModel(self: Config, v: [:0]const u8) void       { gf_config_set_load_model(self.ptr, v.ptr); }
    pub fn setLoadJsonModel(self: Config, v: [:0]const u8) void   { gf_config_set_load_json_model(self.ptr, v.ptr); }
    pub fn setOutputDir(self: Config, v: [:0]const u8) void       { gf_config_set_output_dir(self.ptr, v.ptr); }
    pub fn setDataPath(self: Config, v: [:0]const u8) void        { gf_config_set_data_path(self.ptr, v.ptr); }
    pub fn setAuditLogFile(self: Config, v: [:0]const u8) void    { gf_config_set_audit_log_file(self.ptr, v.ptr); }
    pub fn setEncryptionKey(self: Config, v: [:0]const u8) void   { gf_config_set_encryption_key(self.ptr, v.ptr); }
    pub fn setPatchConfig(self: Config, v: [:0]const u8) void     { gf_config_set_patch_config(self.ptr, v.ptr); }

    /// "relu" | "sigmoid" | "tanh" | "leaky" | "none"
    pub fn setActivation(self: Config, v: [:0]const u8) void  { gf_config_set_activation(self.ptr, v.ptr); }
    /// "gauss" | "uniform" | "analog"
    pub fn setNoiseType(self: Config, v: [:0]const u8) void   { gf_config_set_noise_type(self.ptr, v.ptr); }
    /// "adam" | "sgd" | "rmsprop"
    pub fn setOptimizer(self: Config, v: [:0]const u8) void   { gf_config_set_optimizer(self.ptr, v.ptr); }
    /// "bce" | "wgan" | "hinge" | "ls"
    pub fn setLossType(self: Config, v: [:0]const u8) void    { gf_config_set_loss_type(self.ptr, v.ptr); }
    /// "vector" | "image" | "audio"
    pub fn setDataType(self: Config, v: [:0]const u8) void    { gf_config_set_data_type(self.ptr, v.ptr); }
};

// ── Network ───────────────────────────────────────────────────────────────────

pub const Network = struct {
    ptr: *RawNetwork,

    /// Build a dense generator. sizes: e.g. &[_]c_int{ 64, 128, 1 }.
    pub fn genBuild(sizes: []const c_int, act: [:0]const u8, opt: [:0]const u8, lr: f32) Error!Network {
        return Network{ .ptr = gf_gen_build(sizes.ptr, @intCast(sizes.len), act.ptr, opt.ptr, lr) orelse return error.AllocationFailed };
    }

    /// Build a convolutional generator.
    pub fn genBuildConv(noise_dim: c_int, cond_sz: c_int, base_ch: c_int, act: [:0]const u8, opt: [:0]const u8, lr: f32) Error!Network {
        return Network{ .ptr = gf_gen_build_conv(noise_dim, cond_sz, base_ch, act.ptr, opt.ptr, lr) orelse return error.AllocationFailed };
    }

    /// Build a dense discriminator.
    pub fn discBuild(sizes: []const c_int, act: [:0]const u8, opt: [:0]const u8, lr: f32) Error!Network {
        return Network{ .ptr = gf_disc_build(sizes.ptr, @intCast(sizes.len), act.ptr, opt.ptr, lr) orelse return error.AllocationFailed };
    }

    /// Build a convolutional discriminator.
    pub fn discBuildConv(in_ch: c_int, in_w: c_int, in_h: c_int, cond_sz: c_int, base_ch: c_int, act: [:0]const u8, opt: [:0]const u8, lr: f32) Error!Network {
        return Network{ .ptr = gf_disc_build_conv(in_ch, in_w, in_h, cond_sz, base_ch, act.ptr, opt.ptr, lr) orelse return error.AllocationFailed };
    }

    pub fn deinit(self: Network) void { gf_network_free(self.ptr); }

    pub fn layerCount(self: Network) c_int    { return gf_network_layer_count(self.ptr); }
    pub fn learningRate(self: Network) f32    { return gf_network_learning_rate(self.ptr); }
    pub fn isTraining(self: Network) bool     { return gf_network_is_training(self.ptr) != 0; }

    /// Forward pass. inp: batch×features. Caller owns the result.
    pub fn forward(self: Network, inp: Matrix) Error!Matrix {
        return Matrix{ .ptr = gf_network_forward(self.ptr, inp.ptr) orelse return error.AllocationFailed };
    }
    /// Backward pass. Caller owns the result.
    pub fn backward(self: Network, grad_out: Matrix) Error!Matrix {
        return Matrix{ .ptr = gf_network_backward(self.ptr, grad_out.ptr) orelse return error.AllocationFailed };
    }
    pub fn updateWeights(self: Network) void { gf_network_update_weights(self.ptr); }
    pub fn setTraining(self: Network, training: bool) void {
        gf_network_set_training(self.ptr, ib(training));
    }

    /// Generate count samples. noise_type: "gauss" | "uniform" | "analog".
    pub fn sample(self: Network, count: c_int, noise_dim: c_int, noise_type: [:0]const u8) Error!Matrix {
        return Matrix{ .ptr = gf_network_sample(self.ptr, count, noise_dim, noise_type.ptr) orelse return error.AllocationFailed };
    }

    /// Sanitise weights: replace NaN/Inf with 0.
    pub fn verify(self: Network) void { gf_network_verify(self.ptr); }

    pub fn save(self: Network, path: [:0]const u8) void { gf_network_save(self.ptr, path.ptr); }
    pub fn load(self: Network, path: [:0]const u8) void { gf_network_load(self.ptr, path.ptr); }
};

// ── Dataset ───────────────────────────────────────────────────────────────────

pub const Dataset = struct {
    ptr: *RawDataset,

    pub fn synthetic(n: c_int, features: c_int) Error!Dataset {
        return Dataset{ .ptr = gf_dataset_create_synthetic(n, features) orelse return error.AllocationFailed };
    }

    /// data_type: "vector" | "image" | "audio"
    pub fn load(path: [:0]const u8, data_type: [:0]const u8) Error!Dataset {
        return Dataset{ .ptr = gf_dataset_load(path.ptr, data_type.ptr) orelse return error.AllocationFailed };
    }

    pub fn deinit(self: Dataset) void { gf_dataset_free(self.ptr); }

    pub fn count(self: Dataset) c_int { return gf_dataset_count(self.ptr); }
};

// ── Metrics ───────────────────────────────────────────────────────────────────

pub const Metrics = struct {
    ptr: *RawMetrics,

    pub fn deinit(self: Metrics) void { gf_metrics_free(self.ptr); }

    pub fn dLossReal(self: Metrics) f32   { return gf_metrics_d_loss_real(self.ptr); }
    pub fn dLossFake(self: Metrics) f32   { return gf_metrics_d_loss_fake(self.ptr); }
    pub fn gLoss(self: Metrics) f32       { return gf_metrics_g_loss(self.ptr); }
    pub fn fidScore(self: Metrics) f32    { return gf_metrics_fid_score(self.ptr); }
    pub fn isScore(self: Metrics) f32     { return gf_metrics_is_score(self.ptr); }
    pub fn gradPenalty(self: Metrics) f32 { return gf_metrics_grad_penalty(self.ptr); }
    pub fn epoch(self: Metrics) c_int     { return gf_metrics_epoch(self.ptr); }
    pub fn batch(self: Metrics) c_int     { return gf_metrics_batch(self.ptr); }
};

// ── Result ────────────────────────────────────────────────────────────────────

pub const Result = struct {
    ptr: *RawResult,

    pub fn deinit(self: Result) void { gf_result_free(self.ptr); }

    /// Cloned trained generator. Caller owns and must call deinit().
    pub fn generator(self: Result) Error!Network {
        return Network{ .ptr = gf_result_generator(self.ptr) orelse return error.AllocationFailed };
    }
    /// Cloned trained discriminator. Caller owns and must call deinit().
    pub fn discriminator(self: Result) Error!Network {
        return Network{ .ptr = gf_result_discriminator(self.ptr) orelse return error.AllocationFailed };
    }
    /// Final training metrics. Caller owns and must call deinit().
    pub fn metrics(self: Result) Error!Metrics {
        return Metrics{ .ptr = gf_result_metrics(self.ptr) orelse return error.AllocationFailed };
    }
};

// ── Top-level functions ───────────────────────────────────────────────────────

/// Build networks, train, save — in one call. Caller owns the result.
pub fn run(cfg: *const Config) Error!Result {
    return Result{ .ptr = gf_run(cfg.ptr) orelse return error.AllocationFailed };
}

/// Initialise the global compute backend.
/// name: "cpu" | "cuda" | "opencl" | "hybrid" | "auto"
pub fn initBackend(name: [:0]const u8) void {
    gf_init_backend(name.ptr);
}

/// Detect the best available backend. Returns a static string — do NOT free.
pub fn detectBackend() [:0]const u8 {
    return std.mem.span(gf_detect_backend());
}

/// Seed the global RNG from /dev/urandom or equivalent.
pub fn secureRandomize() void {
    gf_secure_randomize();
}

// ── Training helpers ──────────────────────────────────────────────────────────

pub fn trainFull(gen: *Network, disc: *Network, ds: *const Dataset, cfg: *const Config) Error!Metrics {
    return Metrics{ .ptr = gf_train_full(gen.ptr, disc.ptr, ds.ptr, cfg.ptr) orelse return error.AllocationFailed };
}

pub fn trainStep(gen: *Network, disc: *Network, real_batch: *const Matrix, noise: *const Matrix, cfg: *const Config) Error!Metrics {
    return Metrics{ .ptr = gf_train_step(gen.ptr, disc.ptr, real_batch.ptr, noise.ptr, cfg.ptr) orelse return error.AllocationFailed };
}

pub fn saveJson(gen: *const Network, disc: *const Network, path: [:0]const u8) void {
    gf_train_save_json(gen.ptr, disc.ptr, path.ptr);
}
pub fn loadJson(gen: *Network, disc: *Network, path: [:0]const u8) void {
    gf_train_load_json(gen.ptr, disc.ptr, path.ptr);
}
pub fn saveCheckpoint(gen: *const Network, disc: *const Network, ep: c_int, dir: [:0]const u8) void {
    gf_train_save_checkpoint(gen.ptr, disc.ptr, ep, dir.ptr);
}
pub fn loadCheckpoint(gen: *Network, disc: *Network, ep: c_int, dir: [:0]const u8) void {
    gf_train_load_checkpoint(gen.ptr, disc.ptr, ep, dir.ptr);
}

// ── Loss functions ────────────────────────────────────────────────────────────

pub fn bceLoss(pred: Matrix, target: Matrix) f32 { return gf_bce_loss(pred.ptr, target.ptr); }
pub fn bceGrad(pred: Matrix, target: Matrix) Error!Matrix {
    return Matrix{ .ptr = gf_bce_grad(pred.ptr, target.ptr) orelse return error.AllocationFailed };
}
pub fn wganDiscLoss(d_real: Matrix, d_fake: Matrix) f32 { return gf_wgan_disc_loss(d_real.ptr, d_fake.ptr); }
pub fn wganGenLoss(d_fake: Matrix) f32                  { return gf_wgan_gen_loss(d_fake.ptr); }
pub fn hingeDiscLoss(d_real: Matrix, d_fake: Matrix) f32 { return gf_hinge_disc_loss(d_real.ptr, d_fake.ptr); }
pub fn hingeGenLoss(d_fake: Matrix) f32                 { return gf_hinge_gen_loss(d_fake.ptr); }
pub fn lsDiscLoss(d_real: Matrix, d_fake: Matrix) f32   { return gf_ls_disc_loss(d_real.ptr, d_fake.ptr); }
pub fn lsGenLoss(d_fake: Matrix) f32                    { return gf_ls_gen_loss(d_fake.ptr); }

/// Cosine annealing LR schedule.
pub fn cosineAnneal(ep: c_int, max_ep: c_int, base_lr: f32, min_lr: f32) f32 {
    return gf_cosine_anneal(ep, max_ep, base_lr, min_lr);
}

// ── Random / Noise ────────────────────────────────────────────────────────────

pub fn randomGaussian() f32                        { return gf_random_gaussian(); }
pub fn randomUniform(lo: f32, hi: f32) f32         { return gf_random_uniform(lo, hi); }

/// Generate a size×depth noise matrix. noise_type: "gauss" | "uniform" | "analog".
pub fn generateNoise(size: c_int, depth: c_int, noise_type: [:0]const u8) Error!Matrix {
    return Matrix{ .ptr = gf_generate_noise(size, depth, noise_type.ptr) orelse return error.AllocationFailed };
}

// ── Security ──────────────────────────────────────────────────────────────────

/// Returns true if path is safe (no traversal, etc.).
pub fn validatePath(path: [:0]const u8) bool {
    return gf_validate_path(path.ptr) != 0;
}

/// Append msg to log_file with an ISO-8601 timestamp.
pub fn auditLog(msg: [:0]const u8, log_file: [:0]const u8) void {
    gf_audit_log(msg.ptr, log_file.ptr);
}

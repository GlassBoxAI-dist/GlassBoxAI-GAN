/*
 * facaded_gan.h — C interface to the facaded_gan_cuda GAN library
 *
 * MIT License  Copyright (c) 2025 Matthew Abbott
 *
 * MEMORY RULES
 * ------------
 * Every pointer returned by a gf_* function is heap-allocated and owned by
 * the caller.  You must call the matching gf_*_free() exactly once.
 *
 * Exception: gf_detect_backend() returns a static string — do NOT free it.
 *
 * INPUT CONVENTION
 * ----------------
 * Pointer parameters that are read-only inputs (const GanMatrix*, etc.) are
 * borrowed for the duration of the call.  They remain owned by the caller.
 *
 * THREAD SAFETY
 * -------------
 * The global backend (gf_init_backend / gf_detect_backend) uses a one-shot
 * OnceLock and is safe to call from multiple threads.  Individual GAN objects
 * (GanNetwork, GanConfig, …) are NOT thread-safe; do not share them across
 * threads without external synchronisation.
 *
 * BOOL CONVENTION
 * ---------------
 * Boolean parameters and return values are plain int: 0 = false, ≠0 = true.
 */

#ifndef FACADED_GAN_H
#define FACADED_GAN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>  /* size_t */

/* =========================================================================
 * Opaque handle types
 * ========================================================================= */

/** 2-D matrix of float32 values stored row-major. */
typedef struct GanMatrix_  GanMatrix;

/** 1-D vector of float32 values. */
typedef struct GanVector_  GanVector;

/** Generator or discriminator network. */
typedef struct GanNetwork_ GanNetwork;

/** Training dataset (samples + optional labels). */
typedef struct GanDataset_ GanDataset;

/** Training hyper-parameters and flags. */
typedef struct GanConfig_  GanConfig;

/** Per-step / per-epoch training statistics. */
typedef struct GanMetrics_ GanMetrics;

/** Combined result of gf_run(): trained networks + final metrics. */
typedef struct GanResult_  GanResult;


/* =========================================================================
 * Matrix API
 * ========================================================================= */

/** Create a zero-filled rows×cols matrix.  Free with gf_matrix_free(). */
GanMatrix* gf_matrix_create(int rows, int cols);

/** Copy a caller-supplied flat row-major array into a new matrix.
 *  data[i*cols + j] is row i, column j.  Free with gf_matrix_free(). */
GanMatrix* gf_matrix_from_data(const float* data, int rows, int cols);

/** Free a matrix returned by any gf_* function. */
void gf_matrix_free(GanMatrix* m);

/** Pointer to the internal flat row-major data.  Valid until gf_matrix_free(). */
const float* gf_matrix_data(const GanMatrix* m);

/** Number of rows. */
int   gf_matrix_rows(const GanMatrix* m);

/** Number of columns. */
int   gf_matrix_cols(const GanMatrix* m);

/** Element access (bounds-checked; returns 0 on out-of-range). */
float gf_matrix_get(const GanMatrix* m, int row, int col);

/** Element write (bounds-checked; no-op on out-of-range). */
void  gf_matrix_set(GanMatrix* m, int row, int col, float val);

/** Matrix multiply A×B.  Free result with gf_matrix_free(). */
GanMatrix* gf_matrix_multiply(const GanMatrix* a, const GanMatrix* b);

/** Element-wise addition A+B.  Free result with gf_matrix_free(). */
GanMatrix* gf_matrix_add(const GanMatrix* a, const GanMatrix* b);

/** Element-wise subtraction A−B.  Free result with gf_matrix_free(). */
GanMatrix* gf_matrix_subtract(const GanMatrix* a, const GanMatrix* b);

/** Scalar multiplication.  Free result with gf_matrix_free(). */
GanMatrix* gf_matrix_scale(const GanMatrix* a, float s);

/** Transpose.  Free result with gf_matrix_free(). */
GanMatrix* gf_matrix_transpose(const GanMatrix* a);

/** L2-normalise each row.  Free result with gf_matrix_free(). */
GanMatrix* gf_matrix_normalize(const GanMatrix* a);

/** Element-wise product.  Free result with gf_matrix_free(). */
GanMatrix* gf_matrix_element_mul(const GanMatrix* a, const GanMatrix* b);

/** Safe element read (returns def if out-of-range). */
float gf_matrix_safe_get(const GanMatrix* m, int r, int c, float def);


/* =========================================================================
 * Vector API
 * ========================================================================= */

/** Create a zero-filled vector of length len.  Free with gf_vector_free(). */
GanVector* gf_vector_create(int len);

/** Free a vector returned by any gf_* function. */
void gf_vector_free(GanVector* v);

/** Pointer to the internal data.  Valid until gf_vector_free(). */
const float* gf_vector_data(const GanVector* v);

/** Length of the vector. */
int   gf_vector_len(const GanVector* v);

/** Element access (bounds-checked). */
float gf_vector_get(const GanVector* v, int idx);

/** Spherical linear interpolation between two noise vectors.
 *  t ∈ [0,1].  Free result with gf_vector_free(). */
GanVector* gf_vector_noise_slerp(const GanVector* v1, const GanVector* v2, float t);


/* =========================================================================
 * Config API
 * =========================================================================
 *
 * All fields default to the library's built-in defaults (same as the CLI
 * with no arguments).  Use the setters to override individual fields.
 *
 * Enum-typed fields accept string literals:
 *   activation : "relu" | "sigmoid" | "tanh" | "leaky" | "none"
 *   optimizer  : "adam" | "sgd" | "rmsprop"
 *   loss_type  : "bce"  | "wgan" | "hinge" | "ls"
 *   noise_type : "gauss"| "uniform" | "analog"
 *   data_type  : "vector"| "image" | "audio"
 */

/** Create a config with default values.  Free with gf_config_free(). */
GanConfig* gf_config_create(void);
void       gf_config_free(GanConfig* cfg);

/* Integer fields */
int  gf_config_get_epochs(const GanConfig* c);
void gf_config_set_epochs(GanConfig* c, int v);
int  gf_config_get_batch_size(const GanConfig* c);
void gf_config_set_batch_size(GanConfig* c, int v);
int  gf_config_get_noise_depth(const GanConfig* c);
void gf_config_set_noise_depth(GanConfig* c, int v);
int  gf_config_get_condition_size(const GanConfig* c);
void gf_config_set_condition_size(GanConfig* c, int v);
int  gf_config_get_generator_bits(const GanConfig* c);
void gf_config_set_generator_bits(GanConfig* c, int v);
int  gf_config_get_discriminator_bits(const GanConfig* c);
void gf_config_set_discriminator_bits(GanConfig* c, int v);
int  gf_config_get_max_res_level(const GanConfig* c);
void gf_config_set_max_res_level(GanConfig* c, int v);
int  gf_config_get_metric_interval(const GanConfig* c);
void gf_config_set_metric_interval(GanConfig* c, int v);
int  gf_config_get_checkpoint_interval(const GanConfig* c);
void gf_config_set_checkpoint_interval(GanConfig* c, int v);
int  gf_config_get_fuzz_iterations(const GanConfig* c);
void gf_config_set_fuzz_iterations(GanConfig* c, int v);
int  gf_config_get_num_threads(const GanConfig* c);
void gf_config_set_num_threads(GanConfig* c, int v);

/* Float fields */
float gf_config_get_learning_rate(const GanConfig* c);
void  gf_config_set_learning_rate(GanConfig* c, float v);
float gf_config_get_gp_lambda(const GanConfig* c);
void  gf_config_set_gp_lambda(GanConfig* c, float v);
float gf_config_get_generator_lr(const GanConfig* c);
void  gf_config_set_generator_lr(GanConfig* c, float v);
float gf_config_get_discriminator_lr(const GanConfig* c);
void  gf_config_set_discriminator_lr(GanConfig* c, float v);
float gf_config_get_weight_decay_val(const GanConfig* c);
void  gf_config_set_weight_decay_val(GanConfig* c, float v);

/* Boolean fields (int: 0=false, !=0=true) */
int  gf_config_get_use_batch_norm(const GanConfig* c);
void gf_config_set_use_batch_norm(GanConfig* c, int v);
int  gf_config_get_use_layer_norm(const GanConfig* c);
void gf_config_set_use_layer_norm(GanConfig* c, int v);
int  gf_config_get_use_spectral_norm(const GanConfig* c);
void gf_config_set_use_spectral_norm(GanConfig* c, int v);
int  gf_config_get_use_label_smoothing(const GanConfig* c);
void gf_config_set_use_label_smoothing(GanConfig* c, int v);
int  gf_config_get_use_feature_matching(const GanConfig* c);
void gf_config_set_use_feature_matching(GanConfig* c, int v);
int  gf_config_get_use_minibatch_std_dev(const GanConfig* c);
void gf_config_set_use_minibatch_std_dev(GanConfig* c, int v);
int  gf_config_get_use_progressive(const GanConfig* c);
void gf_config_set_use_progressive(GanConfig* c, int v);
int  gf_config_get_use_augmentation(const GanConfig* c);
void gf_config_set_use_augmentation(GanConfig* c, int v);
int  gf_config_get_compute_metrics(const GanConfig* c);
void gf_config_set_compute_metrics(GanConfig* c, int v);
int  gf_config_get_use_weight_decay(const GanConfig* c);
void gf_config_set_use_weight_decay(GanConfig* c, int v);
int  gf_config_get_use_cosine_anneal(const GanConfig* c);
void gf_config_set_use_cosine_anneal(GanConfig* c, int v);
int  gf_config_get_audit_log(const GanConfig* c);
void gf_config_set_audit_log(GanConfig* c, int v);
int  gf_config_get_use_encryption(const GanConfig* c);
void gf_config_set_use_encryption(GanConfig* c, int v);
int  gf_config_get_use_conv(const GanConfig* c);
void gf_config_set_use_conv(GanConfig* c, int v);
int  gf_config_get_use_attention(const GanConfig* c);
void gf_config_set_use_attention(GanConfig* c, int v);

/* String fields (NUL-terminated UTF-8) */
void gf_config_set_save_model(GanConfig* c, const char* v);
void gf_config_set_load_model(GanConfig* c, const char* v);
void gf_config_set_load_json_model(GanConfig* c, const char* v);
void gf_config_set_output_dir(GanConfig* c, const char* v);
void gf_config_set_data_path(GanConfig* c, const char* v);
void gf_config_set_audit_log_file(GanConfig* c, const char* v);
void gf_config_set_encryption_key(GanConfig* c, const char* v);
void gf_config_set_patch_config(GanConfig* c, const char* v);

/* Enum fields */
void gf_config_set_activation(GanConfig* c, const char* v);
void gf_config_set_noise_type(GanConfig* c, const char* v);
void gf_config_set_optimizer(GanConfig* c, const char* v);
void gf_config_set_loss_type(GanConfig* c, const char* v);
void gf_config_set_data_type(GanConfig* c, const char* v);


/* =========================================================================
 * Network API
 * ========================================================================= */

/** Build a dense generator.  sizes[num_sizes] lists layer widths.
 *  act: activation name.  opt: optimizer name.  lr: learning rate.
 *  Free with gf_network_free(). */
GanNetwork* gf_gen_build(const int* sizes, int num_sizes,
                          const char* act, const char* opt, float lr);

/** Build a convolutional generator.  Free with gf_network_free(). */
GanNetwork* gf_gen_build_conv(int noise_dim, int cond_sz, int base_ch,
                               const char* act, const char* opt, float lr);

/** Build a dense discriminator.  Free with gf_network_free(). */
GanNetwork* gf_disc_build(const int* sizes, int num_sizes,
                           const char* act, const char* opt, float lr);

/** Build a convolutional discriminator.  Free with gf_network_free(). */
GanNetwork* gf_disc_build_conv(int in_ch, int in_w, int in_h,
                                int cond_sz, int base_ch,
                                const char* act, const char* opt, float lr);

/** Free a network returned by any gf_*_build* or gf_result_* function. */
void gf_network_free(GanNetwork* net);

/** Number of layers in the network. */
int   gf_network_layer_count(const GanNetwork* net);

/** Current learning rate. */
float gf_network_learning_rate(const GanNetwork* net);

/** 1 if in training mode, 0 if in inference mode. */
int   gf_network_is_training(const GanNetwork* net);

/** Run a forward pass.  inp is batch_size×input_features.
 *  Free result with gf_matrix_free(). */
GanMatrix* gf_network_forward(GanNetwork* net, const GanMatrix* inp);

/** Run a backward pass.  Free result with gf_matrix_free(). */
GanMatrix* gf_network_backward(GanNetwork* net, const GanMatrix* grad_out);

/** Apply accumulated gradients to weights. */
void gf_network_update_weights(GanNetwork* net);

/** Switch between training (1) and inference (0) mode. */
void gf_network_set_training(GanNetwork* net, int training);

/** Sample count outputs.  noise_type: "gauss","uniform","analog".
 *  Free result with gf_matrix_free(). */
GanMatrix* gf_network_sample(GanNetwork* net, int count,
                              int noise_dim, const char* noise_type);

/** Sanitise weights: replace NaN/Inf with 0. */
void gf_network_verify(GanNetwork* net);

/** Save weights to path (binary JSON). */
void gf_network_save(const GanNetwork* net, const char* path);

/** Load weights from path (binary JSON). */
void gf_network_load(GanNetwork* net, const char* path);


/* =========================================================================
 * Dataset API
 * ========================================================================= */

/** Create a synthetic random dataset.  Free with gf_dataset_free(). */
GanDataset* gf_dataset_create_synthetic(int count, int features);

/** Load a dataset from path.  data_type: "vector","image","audio".
 *  Free with gf_dataset_free(). */
GanDataset* gf_dataset_load(const char* path, const char* data_type);

/** Free a dataset. */
void gf_dataset_free(GanDataset* ds);

/** Number of samples in the dataset. */
int gf_dataset_count(const GanDataset* ds);


/* =========================================================================
 * Training API
 * ========================================================================= */

/** Run all epochs.  Returns final-step metrics.  Free with gf_metrics_free(). */
GanMetrics* gf_train_full(GanNetwork* gen, GanNetwork* disc,
                           const GanDataset* ds, const GanConfig* cfg);

/** Run one discriminator + generator update step.
 *  real_batch: batch×features.  noise: batch×noise_depth.
 *  Free result with gf_metrics_free(). */
GanMetrics* gf_train_step(GanNetwork* gen, GanNetwork* disc,
                           const GanMatrix* real_batch, const GanMatrix* noise,
                           const GanConfig* cfg);

/** Save both networks to a JSON file. */
void gf_train_save_json(const GanNetwork* gen, const GanNetwork* disc,
                         const char* path);

/** Load both networks from a JSON file. */
void gf_train_load_json(GanNetwork* gen, GanNetwork* disc, const char* path);

/** Save a checkpoint (binary) to dir at epoch ep. */
void gf_train_save_checkpoint(const GanNetwork* gen, const GanNetwork* disc,
                               int ep, const char* dir);

/** Load a checkpoint from dir at epoch ep. */
void gf_train_load_checkpoint(GanNetwork* gen, GanNetwork* disc,
                               int ep, const char* dir);


/* =========================================================================
 * Metrics API
 * ========================================================================= */

void  gf_metrics_free(GanMetrics* m);
float gf_metrics_d_loss_real(const GanMetrics* m);
float gf_metrics_d_loss_fake(const GanMetrics* m);
float gf_metrics_g_loss(const GanMetrics* m);
float gf_metrics_fid_score(const GanMetrics* m);
float gf_metrics_is_score(const GanMetrics* m);
float gf_metrics_grad_penalty(const GanMetrics* m);
int   gf_metrics_epoch(const GanMetrics* m);
int   gf_metrics_batch(const GanMetrics* m);


/* =========================================================================
 * High-level Run API
 * ========================================================================= */

/** Build networks, train, save — in one call.  Free with gf_result_free(). */
GanResult* gf_run(const GanConfig* cfg);

void gf_result_free(GanResult* r);

/** Returns a cloned GanNetwork* for the trained generator.
 *  Caller owns it and must call gf_network_free(). */
GanNetwork* gf_result_generator(const GanResult* r);

/** Returns a cloned GanNetwork* for the trained discriminator.
 *  Caller owns it and must call gf_network_free(). */
GanNetwork* gf_result_discriminator(const GanResult* r);

/** Returns a cloned GanMetrics*.  Caller owns it and must call gf_metrics_free(). */
GanMetrics* gf_result_metrics(const GanResult* r);


/* =========================================================================
 * Loss API
 * ========================================================================= */

float      gf_bce_loss(const GanMatrix* pred, const GanMatrix* target);
GanMatrix* gf_bce_grad(const GanMatrix* pred, const GanMatrix* target);
float      gf_wgan_disc_loss(const GanMatrix* d_real, const GanMatrix* d_fake);
float      gf_wgan_gen_loss(const GanMatrix* d_fake);
float      gf_hinge_disc_loss(const GanMatrix* d_real, const GanMatrix* d_fake);
float      gf_hinge_gen_loss(const GanMatrix* d_fake);
float      gf_ls_disc_loss(const GanMatrix* d_real, const GanMatrix* d_fake);
float      gf_ls_gen_loss(const GanMatrix* d_fake);

/** Cosine annealing LR schedule. */
float gf_cosine_anneal(int epoch, int max_ep, float base_lr, float min_lr);


/* =========================================================================
 * Activation API
 * ========================================================================= */

/** Apply a named activation in-place on a copy.  Free with gf_matrix_free().
 *  act_type: "relu","sigmoid","tanh","leaky","none" */
GanMatrix* gf_activate(const GanMatrix* a, const char* act_type);
GanMatrix* gf_relu(const GanMatrix* a);
GanMatrix* gf_sigmoid(const GanMatrix* a);
GanMatrix* gf_tanh_m(const GanMatrix* a);      /* tanh (gf_tanh conflicts with libc) */
GanMatrix* gf_leaky_relu(const GanMatrix* a, float alpha);
GanMatrix* gf_softmax(const GanMatrix* a);


/* =========================================================================
 * Random / Noise API
 * ========================================================================= */

float gf_random_gaussian(void);
float gf_random_uniform(float lo, float hi);

/** Generate a size×depth noise matrix.  noise_type: "gauss","uniform","analog".
 *  Free with gf_matrix_free(). */
GanMatrix* gf_generate_noise(int size, int depth, const char* noise_type);


/* =========================================================================
 * Security API
 * ========================================================================= */

/** Returns 1 if path is safe (no traversal, etc.), else 0. */
int  gf_validate_path(const char* path);

/** Append msg to log_file with an ISO-8601 timestamp. */
void gf_audit_log(const char* msg, const char* log_file);

/** Returns 1 if (r,c) is within matrix bounds, else 0. */
int  gf_bounds_check(const GanMatrix* m, int r, int c);


/* =========================================================================
 * Backend API
 * ========================================================================= */

/** Initialise the global compute backend.
 *  name: "cpu" | "cuda" | "opencl" | "hybrid" | "auto"
 *  Must be called before training if you want a specific backend;
 *  otherwise the library auto-detects on first use. */
void gf_init_backend(const char* name);

/** Detect the best available backend.
 *  Returns a static NUL-terminated string — do NOT free. */
const char* gf_detect_backend(void);

/** Seed the global RNG from /dev/urandom (or equivalent). */
void gf_secure_randomize(void);


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* FACADED_GAN_H */

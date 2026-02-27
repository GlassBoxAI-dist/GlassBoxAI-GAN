/*
 * facaded_gan.hpp — C++ RAII wrapper for the facaded_gan C API
 *
 * MIT License  Copyright (c) 2025 Matthew Abbott
 *
 * USAGE
 * -----
 * Link against libfacaded_gan_c.{so,dylib,dll} or the static variant.
 * Include this header; it includes facaded_gan.h automatically.
 *
 * EXAMPLE
 * -------
 *   using namespace facaded_gan;
 *
 *   Config cfg;
 *   cfg.epochs(2).batch_size(8).use_conv(false);
 *
 *   init_backend("cpu");
 *   auto result = run(cfg);
 *
 *   std::cout << "g_loss: " << result.metrics().g_loss() << "\n";
 *   std::cout << "gen layers: " << result.generator().layer_count() << "\n";
 *
 *   Matrix a(3, 3), b(3, 3);
 *   auto c = a * b;                          // matrix multiply
 *   std::cout << "c(0,0) = " << c(0, 0) << "\n";
 */

#ifndef FACADED_GAN_HPP
#define FACADED_GAN_HPP

#include "facaded_gan.h"

#include <stdexcept>
#include <string>
#include <utility>   /* std::move */
#include <vector>

namespace facaded_gan {

/* =========================================================================
 * Internal helpers
 * ========================================================================= */

namespace detail {
    template<typename T>
    inline T* check(T* p, const char* what = "gf_* allocation failed") {
        if (!p) throw std::runtime_error(what);
        return p;
    }
} // namespace detail

/* =========================================================================
 * Matrix
 * =========================================================================
 *
 * Thin RAII owner of a GanMatrix*.
 * Move-only: copy would require an explicit clone() call.
 */
class Matrix {
public:
    /** Create a zero-filled rows×cols matrix. */
    Matrix(int rows, int cols)
        : p_(detail::check(gf_matrix_create(rows, cols))) {}

    /** Adopt a GanMatrix* returned by the C API.  The Matrix takes ownership. */
    explicit Matrix(GanMatrix* p, bool take_ownership = true)
        : p_(p), owns_(take_ownership) {}

    ~Matrix() { if (owns_ && p_) gf_matrix_free(p_); }

    Matrix(const Matrix&)            = delete;
    Matrix& operator=(const Matrix&) = delete;

    Matrix(Matrix&& o) noexcept : p_(o.p_), owns_(o.owns_) { o.p_ = nullptr; }
    Matrix& operator=(Matrix&& o) noexcept {
        if (this != &o) {
            if (owns_ && p_) gf_matrix_free(p_);
            p_ = o.p_; owns_ = o.owns_; o.p_ = nullptr;
        }
        return *this;
    }

    int          rows()              const { return gf_matrix_rows(p_); }
    int          cols()              const { return gf_matrix_cols(p_); }
    const float* data()              const { return gf_matrix_data(p_); }
    float        get(int r, int c)   const { return gf_matrix_get(p_, r, c); }
    void         set(int r, int c, float v){ gf_matrix_set(p_, r, c, v); }

    /** Element access — read. */
    float operator()(int r, int c) const { return get(r, c); }

    // --- arithmetic (each allocates a new Matrix) ---
    Matrix operator*(const Matrix& b) const {
        return Matrix(detail::check(gf_matrix_multiply(p_, b.p_)));
    }
    Matrix operator+(const Matrix& b) const {
        return Matrix(detail::check(gf_matrix_add(p_, b.p_)));
    }
    Matrix operator-(const Matrix& b) const {
        return Matrix(detail::check(gf_matrix_subtract(p_, b.p_)));
    }
    Matrix operator*(float s) const {
        return Matrix(detail::check(gf_matrix_scale(p_, s)));
    }
    friend Matrix operator*(float s, const Matrix& m) { return m * s; }

    Matrix transpose()  const { return Matrix(detail::check(gf_matrix_transpose(p_))); }
    Matrix normalize()  const { return Matrix(detail::check(gf_matrix_normalize(p_))); }
    Matrix element_mul(const Matrix& b) const {
        return Matrix(detail::check(gf_matrix_element_mul(p_, b.p_)));
    }

    float safe_get(int r, int c, float def = 0.f) const {
        return gf_matrix_safe_get(p_, r, c, def);
    }

    // --- In-place ops ---
    void add_in_place(const Matrix& b)    { gf_matrix_add_in_place(p_, b.p_); }
    void scale_in_place(float s)          { gf_matrix_scale_in_place(p_, s); }
    void clip_in_place(float lo, float hi){ gf_matrix_clip_in_place(p_, lo, hi); }
    void safe_set(int r, int c, float v)  { gf_matrix_safe_set(p_, r, c, v); }

    /** Activation backward pass.  act: "relu","sigmoid","tanh","leaky","none". */
    Matrix activation_backward(const Matrix& pre_act, const char* act) const {
        return Matrix(detail::check(gf_activation_backward(p_, pre_act.p_, act)));
    }

    // --- Activations --- (each returns a new Matrix)
    Matrix relu()                    const { return Matrix(detail::check(gf_relu(p_))); }
    Matrix sigmoid()                 const { return Matrix(detail::check(gf_sigmoid(p_))); }
    Matrix tanh_act()                const { return Matrix(detail::check(gf_tanh_m(p_))); }
    Matrix leaky_relu(float alpha)   const { return Matrix(detail::check(gf_leaky_relu(p_, alpha))); }
    Matrix softmax()                 const { return Matrix(detail::check(gf_softmax(p_))); }
    Matrix activate(const char* act) const { return Matrix(detail::check(gf_activate(p_, act))); }

    /** Raw pointer — use only to pass to C API functions. */
    GanMatrix*       native()       { return p_; }
    const GanMatrix* native() const { return p_; }

private:
    GanMatrix* p_    = nullptr;
    bool       owns_ = true;
};

/* =========================================================================
 * Vector
 * ========================================================================= */
class Vector {
public:
    explicit Vector(int len)
        : p_(detail::check(gf_vector_create(len))) {}
    explicit Vector(GanVector* p) : p_(p) {}

    ~Vector() { if (p_) gf_vector_free(p_); }
    Vector(const Vector&)            = delete;
    Vector& operator=(const Vector&) = delete;
    Vector(Vector&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    Vector& operator=(Vector&& o) noexcept {
        if (this != &o) { if (p_) gf_vector_free(p_); p_ = o.p_; o.p_ = nullptr; }
        return *this;
    }

    int          len()           const { return gf_vector_len(p_); }
    const float* data()          const { return gf_vector_data(p_); }
    float        operator[](int i) const { return gf_vector_get(p_, i); }

    Vector slerp(const Vector& other, float t) const {
        return Vector(detail::check(gf_vector_noise_slerp(p_, other.p_, t)));
    }

    GanVector*       native()       { return p_; }
    const GanVector* native() const { return p_; }

private:
    GanVector* p_ = nullptr;
};

/* =========================================================================
 * Config  (fluent setter API)
 * ========================================================================= */
class Config {
public:
    Config() : p_(detail::check(gf_config_create())) {}
    ~Config() { if (p_) gf_config_free(p_); }
    Config(const Config&)            = delete;
    Config& operator=(const Config&) = delete;
    Config(Config&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    Config& operator=(Config&& o) noexcept {
        if (this != &o) { if (p_) gf_config_free(p_); p_ = o.p_; o.p_ = nullptr; }
        return *this;
    }

    // --- fluent setters (return *this for chaining) ---
    Config& epochs(int v)              { gf_config_set_epochs(p_, v);              return *this; }
    Config& batch_size(int v)          { gf_config_set_batch_size(p_, v);          return *this; }
    Config& noise_depth(int v)         { gf_config_set_noise_depth(p_, v);         return *this; }
    Config& condition_size(int v)      { gf_config_set_condition_size(p_, v);      return *this; }
    Config& generator_bits(int v)      { gf_config_set_generator_bits(p_, v);      return *this; }
    Config& discriminator_bits(int v)  { gf_config_set_discriminator_bits(p_, v);  return *this; }
    Config& max_res_level(int v)       { gf_config_set_max_res_level(p_, v);       return *this; }
    Config& metric_interval(int v)     { gf_config_set_metric_interval(p_, v);     return *this; }
    Config& checkpoint_interval(int v) { gf_config_set_checkpoint_interval(p_, v); return *this; }
    Config& fuzz_iterations(int v)     { gf_config_set_fuzz_iterations(p_, v);     return *this; }
    Config& num_threads(int v)         { gf_config_set_num_threads(p_, v);         return *this; }

    Config& learning_rate(float v)     { gf_config_set_learning_rate(p_, v);       return *this; }
    Config& gp_lambda(float v)         { gf_config_set_gp_lambda(p_, v);           return *this; }
    Config& generator_lr(float v)      { gf_config_set_generator_lr(p_, v);        return *this; }
    Config& discriminator_lr(float v)  { gf_config_set_discriminator_lr(p_, v);    return *this; }
    Config& weight_decay_val(float v)  { gf_config_set_weight_decay_val(p_, v);    return *this; }

    Config& use_batch_norm(bool v)       { gf_config_set_use_batch_norm(p_, v);       return *this; }
    Config& use_layer_norm(bool v)       { gf_config_set_use_layer_norm(p_, v);       return *this; }
    Config& use_spectral_norm(bool v)    { gf_config_set_use_spectral_norm(p_, v);    return *this; }
    Config& use_label_smoothing(bool v)  { gf_config_set_use_label_smoothing(p_, v);  return *this; }
    Config& use_feature_matching(bool v) { gf_config_set_use_feature_matching(p_, v); return *this; }
    Config& use_minibatch_std_dev(bool v){ gf_config_set_use_minibatch_std_dev(p_,v); return *this; }
    Config& use_progressive(bool v)      { gf_config_set_use_progressive(p_, v);      return *this; }
    Config& use_augmentation(bool v)     { gf_config_set_use_augmentation(p_, v);     return *this; }
    Config& compute_metrics(bool v)      { gf_config_set_compute_metrics(p_, v);      return *this; }
    Config& use_weight_decay(bool v)     { gf_config_set_use_weight_decay(p_, v);     return *this; }
    Config& use_cosine_anneal(bool v)    { gf_config_set_use_cosine_anneal(p_, v);    return *this; }
    Config& audit_log(bool v)            { gf_config_set_audit_log(p_, v);            return *this; }
    Config& use_encryption(bool v)       { gf_config_set_use_encryption(p_, v);       return *this; }
    Config& use_conv(bool v)             { gf_config_set_use_conv(p_, v);             return *this; }
    Config& use_attention(bool v)        { gf_config_set_use_attention(p_, v);        return *this; }

    Config& save_model(const std::string& v)      { gf_config_set_save_model(p_,      v.c_str()); return *this; }
    Config& load_model(const std::string& v)      { gf_config_set_load_model(p_,      v.c_str()); return *this; }
    Config& load_json_model(const std::string& v) { gf_config_set_load_json_model(p_, v.c_str()); return *this; }
    Config& output_dir(const std::string& v)      { gf_config_set_output_dir(p_,      v.c_str()); return *this; }
    Config& data_path(const std::string& v)       { gf_config_set_data_path(p_,       v.c_str()); return *this; }
    Config& audit_log_file(const std::string& v)  { gf_config_set_audit_log_file(p_,  v.c_str()); return *this; }
    Config& encryption_key(const std::string& v)  { gf_config_set_encryption_key(p_,  v.c_str()); return *this; }
    Config& patch_config(const std::string& v)    { gf_config_set_patch_config(p_,    v.c_str()); return *this; }

    Config& activation(const char* v)  { gf_config_set_activation(p_, v);  return *this; }
    Config& noise_type(const char* v)  { gf_config_set_noise_type(p_, v);  return *this; }
    Config& optimizer(const char* v)   { gf_config_set_optimizer(p_, v);   return *this; }
    Config& loss_type(const char* v)   { gf_config_set_loss_type(p_, v);   return *this; }
    Config& data_type(const char* v)   { gf_config_set_data_type(p_, v);   return *this; }

    // --- getters ---
    int   get_epochs()        const { return gf_config_get_epochs(p_); }
    int   get_batch_size()    const { return gf_config_get_batch_size(p_); }
    float get_learning_rate() const { return gf_config_get_learning_rate(p_); }

    GanConfig*       native()       { return p_; }
    const GanConfig* native() const { return p_; }

private:
    GanConfig* p_ = nullptr;
};

/* =========================================================================
 * Network
 * ========================================================================= */
class Network {
public:
    /** Adopt an externally allocated GanNetwork* (e.g. from gf_result_generator). */
    explicit Network(GanNetwork* p) : p_(detail::check(p)) {}

    /** Build a dense network with the given layer sizes. */
    static Network gen_dense(const std::vector<int>& sizes,
                              const char* act = "leaky",
                              const char* opt = "adam",
                              float lr = 0.0002f) {
        return Network(detail::check(
            gf_gen_build(sizes.data(), static_cast<int>(sizes.size()), act, opt, lr)));
    }
    static Network disc_dense(const std::vector<int>& sizes,
                               const char* act = "leaky",
                               const char* opt = "adam",
                               float lr = 0.0002f) {
        return Network(detail::check(
            gf_disc_build(sizes.data(), static_cast<int>(sizes.size()), act, opt, lr)));
    }
    static Network gen_conv(int noise_dim, int cond_sz = 0, int base_ch = 8,
                             const char* act = "leaky",
                             const char* opt = "adam",
                             float lr = 0.0002f) {
        return Network(detail::check(
            gf_gen_build_conv(noise_dim, cond_sz, base_ch, act, opt, lr)));
    }
    static Network disc_conv(int in_ch, int in_w, int in_h,
                              int cond_sz = 0, int base_ch = 8,
                              const char* act = "leaky",
                              const char* opt = "adam",
                              float lr = 0.0002f) {
        return Network(detail::check(
            gf_disc_build_conv(in_ch, in_w, in_h, cond_sz, base_ch, act, opt, lr)));
    }

    ~Network() { if (p_) gf_network_free(p_); }
    Network(const Network&)            = delete;
    Network& operator=(const Network&) = delete;
    Network(Network&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    Network& operator=(Network&& o) noexcept {
        if (this != &o) { if (p_) gf_network_free(p_); p_ = o.p_; o.p_ = nullptr; }
        return *this;
    }

    int   layer_count()    const { return gf_network_layer_count(p_); }
    float learning_rate()  const { return gf_network_learning_rate(p_); }
    bool  is_training()    const { return gf_network_is_training(p_) != 0; }

    Matrix forward(const Matrix& inp)       { return Matrix(detail::check(gf_network_forward(p_, inp.native()))); }
    Matrix backward(const Matrix& grad_out) { return Matrix(detail::check(gf_network_backward(p_, grad_out.native()))); }
    void   update_weights()                 { gf_network_update_weights(p_); }
    void   set_training(bool t)             { gf_network_set_training(p_, t ? 1 : 0); }

    Matrix sample(int count, int noise_dim, const char* noise_type = "gauss") {
        return Matrix(detail::check(gf_network_sample(p_, count, noise_dim, noise_type)));
    }

    void verify()                       { gf_network_verify(p_); }
    void save(const std::string& path)  { gf_network_save(p_, path.c_str()); }
    void load(const std::string& path)  { gf_network_load(p_, path.c_str()); }

    // --- Generator extensions ---
    Matrix sample_conditional(int count, int noise_dim, int cond_sz,
                               const char* noise_type, const Matrix& cond) {
        return Matrix(detail::check(
            gf_gen_sample_conditional(p_, count, noise_dim, cond_sz,
                                      noise_type, cond.native())));
    }
    void   add_progressive_layer(int res_lvl) { gf_gen_add_progressive_layer(p_, res_lvl); }
    Matrix get_layer_output(int idx) const {
        return Matrix(detail::check(gf_gen_get_layer_output(p_, idx)));
    }
    Network deep_copy() const {
        return Network(detail::check(gf_gen_deep_copy(p_)));
    }

    // --- Discriminator extensions ---
    Matrix disc_evaluate(const Matrix& inp) {
        return Matrix(detail::check(gf_disc_evaluate(p_, inp.native())));
    }
    float disc_grad_penalty(const Matrix& real, const Matrix& fake, float lambda) {
        return gf_disc_grad_penalty(p_, real.native(), fake.native(), lambda);
    }
    float disc_feature_match(const Matrix& real, const Matrix& fake, int feat_layer) {
        return gf_disc_feature_match(p_, real.native(), fake.native(), feat_layer);
    }
    void disc_add_progressive_layer(int res_lvl) {
        gf_disc_add_progressive_layer(p_, res_lvl);
    }
    Matrix disc_get_layer_output(int idx) const {
        return Matrix(detail::check(gf_disc_get_layer_output(p_, idx)));
    }
    Network disc_deep_copy() const {
        return Network(detail::check(gf_disc_deep_copy(p_)));
    }

    // --- Optimizer shortcut ---
    void optimize() { gf_train_optimize(p_); }

    GanNetwork*       native()       { return p_; }
    const GanNetwork* native() const { return p_; }

private:
    GanNetwork* p_ = nullptr;
};

/* =========================================================================
 * Dataset
 * ========================================================================= */
class Dataset {
public:
    /** Create a synthetic random dataset. */
    static Dataset synthetic(int count, int features) {
        return Dataset(detail::check(gf_dataset_create_synthetic(count, features)));
    }
    static Dataset load(const std::string& path, const char* data_type = "vector") {
        return Dataset(detail::check(gf_dataset_load(path.c_str(), data_type)));
    }
    static Dataset load_bmp(const std::string& path) {
        return Dataset(detail::check(gf_train_load_bmp(path.c_str())));
    }
    static Dataset load_wav(const std::string& path) {
        return Dataset(detail::check(gf_train_load_wav(path.c_str())));
    }

    explicit Dataset(GanDataset* p) : p_(p) {}
    ~Dataset() { if (p_) gf_dataset_free(p_); }
    Dataset(const Dataset&)            = delete;
    Dataset& operator=(const Dataset&) = delete;
    Dataset(Dataset&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    Dataset& operator=(Dataset&& o) noexcept {
        if (this != &o) { if (p_) gf_dataset_free(p_); p_ = o.p_; o.p_ = nullptr; }
        return *this;
    }

    int count() const { return gf_dataset_count(p_); }

    GanDataset*       native()       { return p_; }
    const GanDataset* native() const { return p_; }

private:
    GanDataset* p_ = nullptr;
};

/* =========================================================================
 * Metrics  (value-like wrapper — no free needed from user perspective)
 * ========================================================================= */
class Metrics {
public:
    explicit Metrics(GanMetrics* p) : p_(detail::check(p)) {}
    ~Metrics() { if (p_) gf_metrics_free(p_); }
    Metrics(const Metrics&)            = delete;
    Metrics& operator=(const Metrics&) = delete;
    Metrics(Metrics&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    Metrics& operator=(Metrics&& o) noexcept {
        if (this != &o) { if (p_) gf_metrics_free(p_); p_ = o.p_; o.p_ = nullptr; }
        return *this;
    }

    float d_loss_real()  const { return gf_metrics_d_loss_real(p_); }
    float d_loss_fake()  const { return gf_metrics_d_loss_fake(p_); }
    float g_loss()       const { return gf_metrics_g_loss(p_); }
    float fid_score()    const { return gf_metrics_fid_score(p_); }
    float is_score()     const { return gf_metrics_is_score(p_); }
    float grad_penalty() const { return gf_metrics_grad_penalty(p_); }
    int   epoch()        const { return gf_metrics_epoch(p_); }
    int   batch()        const { return gf_metrics_batch(p_); }

    GanMetrics*       native()       { return p_; }
    const GanMetrics* native() const { return p_; }

private:
    GanMetrics* p_ = nullptr;
};

/* =========================================================================
 * Result
 * ========================================================================= */
class Result {
public:
    explicit Result(GanResult* p) : p_(detail::check(p)) {}
    ~Result() { if (p_) gf_result_free(p_); }
    Result(const Result&)            = delete;
    Result& operator=(const Result&) = delete;
    Result(Result&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    Result& operator=(Result&& o) noexcept {
        if (this != &o) { if (p_) gf_result_free(p_); p_ = o.p_; o.p_ = nullptr; }
        return *this;
    }

    /** Cloned generator network — caller owns the returned Network. */
    Network generator()     { return Network(detail::check(gf_result_generator(p_))); }

    /** Cloned discriminator network — caller owns the returned Network. */
    Network discriminator() { return Network(detail::check(gf_result_discriminator(p_))); }

    /** Cloned metrics — caller owns the returned Metrics. */
    Metrics metrics()       { return Metrics(detail::check(gf_result_metrics(p_))); }

private:
    GanResult* p_ = nullptr;
};

/* =========================================================================
 * MatrixArray  (for FID / IS metrics)
 * ========================================================================= */
class MatrixArray {
public:
    MatrixArray() : p_(detail::check(gf_matrix_array_create())) {}
    explicit MatrixArray(GanMatrixArray* p) : p_(p) {}
    ~MatrixArray() { if (p_) gf_matrix_array_free(p_); }
    MatrixArray(const MatrixArray&)            = delete;
    MatrixArray& operator=(const MatrixArray&) = delete;
    MatrixArray(MatrixArray&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    MatrixArray& operator=(MatrixArray&& o) noexcept {
        if (this != &o) { if (p_) gf_matrix_array_free(p_); p_ = o.p_; o.p_ = nullptr; }
        return *this;
    }

    void push(const Matrix& m) { gf_matrix_array_push(p_, m.native()); }
    int  len()           const { return gf_matrix_array_len(p_); }

    GanMatrixArray*       native()       { return p_; }
    const GanMatrixArray* native() const { return p_; }

private:
    GanMatrixArray* p_ = nullptr;
};

/* =========================================================================
 * Layer  (GanLayer RAII wrapper)
 * ========================================================================= */
class Layer {
public:
    /** Adopt an existing GanLayer* (takes ownership). */
    explicit Layer(GanLayer* p) : p_(detail::check(p)) {}
    ~Layer() { if (p_) gf_layer_free(p_); }
    Layer(const Layer&)            = delete;
    Layer& operator=(const Layer&) = delete;
    Layer(Layer&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    Layer& operator=(Layer&& o) noexcept {
        if (this != &o) { if (p_) gf_layer_free(p_); p_ = o.p_; o.p_ = nullptr; }
        return *this;
    }

    // --- factories ---
    static Layer dense(int in_sz, int out_sz, const char* act = "relu") {
        return Layer(detail::check(gf_layer_create_dense(in_sz, out_sz, act)));
    }
    static Layer conv2d(int in_ch, int out_ch, int k, int s, int pad,
                        int w, int h, const char* act = "leaky") {
        return Layer(detail::check(gf_layer_create_conv2d(in_ch, out_ch, k, s, pad, w, h, act)));
    }
    static Layer deconv2d(int in_ch, int out_ch, int k, int s, int pad,
                          int w, int h, const char* act = "relu") {
        return Layer(detail::check(gf_layer_create_deconv2d(in_ch, out_ch, k, s, pad, w, h, act)));
    }
    static Layer conv1d(int in_ch, int out_ch, int k, int s, int pad,
                        int in_len, const char* act = "leaky") {
        return Layer(detail::check(gf_layer_create_conv1d(in_ch, out_ch, k, s, pad, in_len, act)));
    }
    static Layer batch_norm(int features) {
        return Layer(detail::check(gf_layer_create_batch_norm(features)));
    }
    static Layer layer_norm(int features) {
        return Layer(detail::check(gf_layer_create_layer_norm(features)));
    }
    static Layer attention(int d_model, int n_heads) {
        return Layer(detail::check(gf_layer_create_attention(d_model, n_heads)));
    }

    // --- dispatch ---
    Matrix forward(const Matrix& inp)  { return Matrix(detail::check(gf_layer_forward(p_, inp.native()))); }
    Matrix backward(const Matrix& grad){ return Matrix(detail::check(gf_layer_backward(p_, grad.native()))); }
    void   init_optimizer(const char* opt = "adam") { gf_layer_init_optimizer(p_, opt); }

    // --- specific ops ---
    Matrix conv2d_fwd(const Matrix& inp)        { return Matrix(detail::check(gf_layer_conv2d(inp.native(), p_))); }
    Matrix conv2d_bwd(const Matrix& grad)       { return Matrix(detail::check(gf_layer_conv2d_backward(p_, grad.native()))); }
    Matrix deconv2d_fwd(const Matrix& inp)      { return Matrix(detail::check(gf_layer_deconv2d(inp.native(), p_))); }
    Matrix deconv2d_bwd(const Matrix& grad)     { return Matrix(detail::check(gf_layer_deconv2d_backward(p_, grad.native()))); }
    Matrix conv1d_fwd(const Matrix& inp)        { return Matrix(detail::check(gf_layer_conv1d(inp.native(), p_))); }
    Matrix conv1d_bwd(const Matrix& grad)       { return Matrix(detail::check(gf_layer_conv1d_backward(p_, grad.native()))); }
    Matrix batch_norm_fwd(const Matrix& inp)    { return Matrix(detail::check(gf_layer_batch_norm(inp.native(), p_))); }
    Matrix batch_norm_bwd(const Matrix& grad)   { return Matrix(detail::check(gf_layer_batch_norm_backward(p_, grad.native()))); }
    Matrix layer_norm_fwd(const Matrix& inp)    { return Matrix(detail::check(gf_layer_layer_norm(inp.native(), p_))); }
    Matrix layer_norm_bwd(const Matrix& grad)   { return Matrix(detail::check(gf_layer_layer_norm_backward(p_, grad.native()))); }
    Matrix spectral_norm()                      { return Matrix(detail::check(gf_layer_spectral_norm(p_))); }
    Matrix attention_fwd(const Matrix& inp)     { return Matrix(detail::check(gf_layer_attention(inp.native(), p_))); }
    Matrix attention_bwd(const Matrix& grad)    { return Matrix(detail::check(gf_layer_attention_backward(p_, grad.native()))); }
    void   verify_weights()                     { gf_layer_verify_weights(p_); }

    GanLayer*       native()       { return p_; }
    const GanLayer* native() const { return p_; }

private:
    GanLayer* p_ = nullptr;
};

/* =========================================================================
 * Free functions
 * ========================================================================= */

/** Initialise the global compute backend ("cpu","cuda","opencl","hybrid","auto"). */
inline void init_backend(const char* name) { gf_init_backend(name); }

/** Returns the name of the best detected backend. */
inline std::string detect_backend() { return gf_detect_backend(); }

/** Seed the RNG from /dev/urandom. */
inline void secure_randomize() { gf_secure_randomize(); }

/** Full pipeline: build, train, return Result. */
inline Result run(Config& cfg) {
    return Result(detail::check(gf_run(cfg.native())));
}

/** Manual training: run all epochs. */
inline Metrics train_full(Network& gen, Network& disc, Dataset& ds, Config& cfg) {
    return Metrics(detail::check(
        gf_train_full(gen.native(), disc.native(), ds.native(), cfg.native())));
}

/** Manual training: single step. */
inline Metrics train_step(Network& gen, Network& disc,
                           const Matrix& real_batch, const Matrix& noise,
                           Config& cfg) {
    return Metrics(detail::check(
        gf_train_step(gen.native(), disc.native(),
                      real_batch.native(), noise.native(), cfg.native())));
}

/** Save both networks to a JSON file. */
inline void save_json(const Network& gen, const Network& disc,
                      const std::string& path) {
    gf_train_save_json(gen.native(), disc.native(), path.c_str());
}

/** Load both networks from a JSON file. */
inline void load_json(Network& gen, Network& disc, const std::string& path) {
    gf_train_load_json(gen.native(), disc.native(), path.c_str());
}

/** Generate a size×depth noise matrix (row per sample). */
inline Matrix generate_noise(int size, int depth,
                              const char* noise_type = "gauss") {
    return Matrix(detail::check(gf_generate_noise(size, depth, noise_type)));
}

// --- Loss free functions ---
inline float bce_loss(const Matrix& pred, const Matrix& target) {
    return gf_bce_loss(pred.native(), target.native());
}
inline float wgan_disc_loss(const Matrix& d_real, const Matrix& d_fake) {
    return gf_wgan_disc_loss(d_real.native(), d_fake.native());
}
inline float wgan_gen_loss(const Matrix& d_fake) {
    return gf_wgan_gen_loss(d_fake.native());
}
inline float hinge_disc_loss(const Matrix& d_real, const Matrix& d_fake) {
    return gf_hinge_disc_loss(d_real.native(), d_fake.native());
}
inline float hinge_gen_loss(const Matrix& d_fake) {
    return gf_hinge_gen_loss(d_fake.native());
}
inline float ls_disc_loss(const Matrix& d_real, const Matrix& d_fake) {
    return gf_ls_disc_loss(d_real.native(), d_fake.native());
}
inline float ls_gen_loss(const Matrix& d_fake) {
    return gf_ls_gen_loss(d_fake.native());
}
inline float cosine_anneal(int epoch, int max_ep,
                            float base_lr, float min_lr) {
    return gf_cosine_anneal(epoch, max_ep, base_lr, min_lr);
}

// --- Random ---
inline float random_gaussian() { return gf_random_gaussian(); }
inline float random_uniform(float lo, float hi) { return gf_random_uniform(lo, hi); }

// --- Security ---
inline bool validate_path(const std::string& path) {
    return gf_validate_path(path.c_str()) != 0;
}
inline void audit_log(const std::string& msg, const std::string& file) {
    gf_audit_log(msg.c_str(), file.c_str());
}
inline bool bounds_check(const Matrix& m, int r, int c) {
    return gf_bounds_check(m.native(), r, c) != 0;
}

// --- Security extensions ---
inline unsigned char sec_get_os_random() { return gf_sec_get_os_random(); }
inline void sec_encrypt_model(const std::string& in_f, const std::string& out_f,
                               const std::string& key) {
    gf_sec_encrypt_model(in_f.c_str(), out_f.c_str(), key.c_str());
}
inline void sec_decrypt_model(const std::string& in_f, const std::string& out_f,
                               const std::string& key) {
    gf_sec_decrypt_model(in_f.c_str(), out_f.c_str(), key.c_str());
}
inline bool sec_run_tests()                    { return gf_sec_run_tests() != 0; }
inline bool sec_run_fuzz_tests(int iterations) { return gf_sec_run_fuzz_tests(iterations) != 0; }

// --- Training extensions ---
inline Matrix label_smoothing(const Matrix& labels, float lo, float hi) {
    return Matrix(detail::check(gf_train_label_smoothing(labels.native(), lo, hi)));
}
inline Matrix disc_minibatch_std_dev(const Matrix& inp) {
    return Matrix(detail::check(gf_disc_minibatch_std_dev(inp.native())));
}
inline void train_log_metrics(Metrics& m, const std::string& filename) {
    gf_train_log_metrics(m.native(), filename.c_str());
}
inline void train_print_bar(float d_loss, float g_loss, int width = 40) {
    gf_train_print_bar(d_loss, g_loss, width);
}
inline float compute_fid(const MatrixArray& real_arr, const MatrixArray& fake_arr) {
    return gf_train_compute_fid(real_arr.native(), fake_arr.native());
}
inline float compute_is(const MatrixArray& samples) {
    return gf_train_compute_is(samples.native());
}

} // namespace facaded_gan

#endif /* FACADED_GAN_HPP */

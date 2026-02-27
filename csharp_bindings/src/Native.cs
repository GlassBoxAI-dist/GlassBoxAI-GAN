// MIT License  Copyright (c) 2025 Matthew Abbott
// P/Invoke declarations for libfacaded_gan_c.{so,dylib,dll}
// All string parameters are NUL-terminated UTF-8 (Ansi charset is UTF-8 on Linux/macOS).

using System.Runtime.InteropServices;

namespace FacadedGan;

internal static class Native
{
    private const string Lib = "facaded_gan_c";

    // ─── Matrix ───────────────────────────────────────────────────────────────
    [DllImport(Lib)] internal static extern IntPtr gf_matrix_create(int rows, int cols);
    [DllImport(Lib)] internal static extern IntPtr gf_matrix_from_data([In] float[] data, int rows, int cols);
    [DllImport(Lib)] internal static extern void   gf_matrix_free(IntPtr m);
    [DllImport(Lib)] internal static extern IntPtr gf_matrix_data(IntPtr m);
    [DllImport(Lib)] internal static extern int    gf_matrix_rows(IntPtr m);
    [DllImport(Lib)] internal static extern int    gf_matrix_cols(IntPtr m);
    [DllImport(Lib)] internal static extern float  gf_matrix_get(IntPtr m, int row, int col);
    [DllImport(Lib)] internal static extern void   gf_matrix_set(IntPtr m, int row, int col, float val);
    [DllImport(Lib)] internal static extern IntPtr gf_matrix_multiply(IntPtr a, IntPtr b);
    [DllImport(Lib)] internal static extern IntPtr gf_matrix_add(IntPtr a, IntPtr b);
    [DllImport(Lib)] internal static extern IntPtr gf_matrix_subtract(IntPtr a, IntPtr b);
    [DllImport(Lib)] internal static extern IntPtr gf_matrix_scale(IntPtr a, float s);
    [DllImport(Lib)] internal static extern IntPtr gf_matrix_transpose(IntPtr a);
    [DllImport(Lib)] internal static extern IntPtr gf_matrix_normalize(IntPtr a);
    [DllImport(Lib)] internal static extern IntPtr gf_matrix_element_mul(IntPtr a, IntPtr b);
    [DllImport(Lib)] internal static extern float  gf_matrix_safe_get(IntPtr m, int r, int c, float def);
    [DllImport(Lib)] internal static extern int    gf_bounds_check(IntPtr m, int r, int c);

    // ─── Vector ───────────────────────────────────────────────────────────────
    [DllImport(Lib)] internal static extern IntPtr gf_vector_create(int len);
    [DllImport(Lib)] internal static extern void   gf_vector_free(IntPtr v);
    [DllImport(Lib)] internal static extern IntPtr gf_vector_data(IntPtr v);
    [DllImport(Lib)] internal static extern int    gf_vector_len(IntPtr v);
    [DllImport(Lib)] internal static extern float  gf_vector_get(IntPtr v, int idx);
    [DllImport(Lib)] internal static extern IntPtr gf_vector_noise_slerp(IntPtr v1, IntPtr v2, float t);

    // ─── Config ───────────────────────────────────────────────────────────────
    [DllImport(Lib)] internal static extern IntPtr gf_config_create();
    [DllImport(Lib)] internal static extern void   gf_config_free(IntPtr c);

    [DllImport(Lib)] internal static extern int  gf_config_get_epochs(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_epochs(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_batch_size(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_batch_size(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_noise_depth(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_noise_depth(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_condition_size(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_condition_size(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_generator_bits(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_generator_bits(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_discriminator_bits(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_discriminator_bits(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_max_res_level(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_max_res_level(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_metric_interval(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_metric_interval(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_checkpoint_interval(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_checkpoint_interval(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_fuzz_iterations(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_fuzz_iterations(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_num_threads(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_num_threads(IntPtr c, int v);

    [DllImport(Lib)] internal static extern float gf_config_get_learning_rate(IntPtr c);
    [DllImport(Lib)] internal static extern void  gf_config_set_learning_rate(IntPtr c, float v);
    [DllImport(Lib)] internal static extern float gf_config_get_gp_lambda(IntPtr c);
    [DllImport(Lib)] internal static extern void  gf_config_set_gp_lambda(IntPtr c, float v);
    [DllImport(Lib)] internal static extern float gf_config_get_generator_lr(IntPtr c);
    [DllImport(Lib)] internal static extern void  gf_config_set_generator_lr(IntPtr c, float v);
    [DllImport(Lib)] internal static extern float gf_config_get_discriminator_lr(IntPtr c);
    [DllImport(Lib)] internal static extern void  gf_config_set_discriminator_lr(IntPtr c, float v);
    [DllImport(Lib)] internal static extern float gf_config_get_weight_decay_val(IntPtr c);
    [DllImport(Lib)] internal static extern void  gf_config_set_weight_decay_val(IntPtr c, float v);

    [DllImport(Lib)] internal static extern int  gf_config_get_use_batch_norm(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_use_batch_norm(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_use_layer_norm(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_use_layer_norm(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_use_spectral_norm(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_use_spectral_norm(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_use_label_smoothing(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_use_label_smoothing(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_use_feature_matching(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_use_feature_matching(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_use_minibatch_std_dev(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_use_minibatch_std_dev(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_use_progressive(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_use_progressive(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_use_augmentation(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_use_augmentation(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_compute_metrics(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_compute_metrics(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_use_weight_decay(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_use_weight_decay(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_use_cosine_anneal(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_use_cosine_anneal(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_audit_log(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_audit_log(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_use_encryption(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_use_encryption(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_use_conv(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_use_conv(IntPtr c, int v);
    [DllImport(Lib)] internal static extern int  gf_config_get_use_attention(IntPtr c);
    [DllImport(Lib)] internal static extern void gf_config_set_use_attention(IntPtr c, int v);

    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_config_set_save_model(IntPtr c, string v);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_config_set_load_model(IntPtr c, string v);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_config_set_load_json_model(IntPtr c, string v);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_config_set_output_dir(IntPtr c, string v);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_config_set_data_path(IntPtr c, string v);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_config_set_audit_log_file(IntPtr c, string v);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_config_set_encryption_key(IntPtr c, string v);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_config_set_patch_config(IntPtr c, string v);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_config_set_activation(IntPtr c, string v);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_config_set_noise_type(IntPtr c, string v);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_config_set_optimizer(IntPtr c, string v);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_config_set_loss_type(IntPtr c, string v);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_config_set_data_type(IntPtr c, string v);

    // ─── Network ──────────────────────────────────────────────────────────────
    [DllImport(Lib, CharSet = CharSet.Ansi)]
    internal static extern IntPtr gf_gen_build([In] int[] sizes, int numSizes, string act, string opt, float lr);
    [DllImport(Lib, CharSet = CharSet.Ansi)]
    internal static extern IntPtr gf_gen_build_conv(int noiseDim, int condSz, int baseCh, string act, string opt, float lr);
    [DllImport(Lib, CharSet = CharSet.Ansi)]
    internal static extern IntPtr gf_disc_build([In] int[] sizes, int numSizes, string act, string opt, float lr);
    [DllImport(Lib, CharSet = CharSet.Ansi)]
    internal static extern IntPtr gf_disc_build_conv(int inCh, int inW, int inH, int condSz, int baseCh, string act, string opt, float lr);
    [DllImport(Lib)] internal static extern void   gf_network_free(IntPtr net);
    [DllImport(Lib)] internal static extern int    gf_network_layer_count(IntPtr net);
    [DllImport(Lib)] internal static extern float  gf_network_learning_rate(IntPtr net);
    [DllImport(Lib)] internal static extern int    gf_network_is_training(IntPtr net);
    [DllImport(Lib)] internal static extern IntPtr gf_network_forward(IntPtr net, IntPtr inp);
    [DllImport(Lib)] internal static extern IntPtr gf_network_backward(IntPtr net, IntPtr gradOut);
    [DllImport(Lib)] internal static extern void   gf_network_update_weights(IntPtr net);
    [DllImport(Lib)] internal static extern void   gf_network_set_training(IntPtr net, int training);
    [DllImport(Lib, CharSet = CharSet.Ansi)]
    internal static extern IntPtr gf_network_sample(IntPtr net, int count, int noiseDim, string noiseType);
    [DllImport(Lib)] internal static extern void   gf_network_verify(IntPtr net);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_network_save(IntPtr net, string path);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_network_load(IntPtr net, string path);

    // ─── Dataset ──────────────────────────────────────────────────────────────
    [DllImport(Lib)] internal static extern IntPtr gf_dataset_create_synthetic(int count, int features);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern IntPtr gf_dataset_load(string path, string dataType);
    [DllImport(Lib)] internal static extern void   gf_dataset_free(IntPtr ds);
    [DllImport(Lib)] internal static extern int    gf_dataset_count(IntPtr ds);

    // ─── Training ─────────────────────────────────────────────────────────────
    [DllImport(Lib)] internal static extern IntPtr gf_train_full(IntPtr gen, IntPtr disc, IntPtr ds, IntPtr cfg);
    [DllImport(Lib)] internal static extern IntPtr gf_train_step(IntPtr gen, IntPtr disc, IntPtr realBatch, IntPtr noise, IntPtr cfg);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_train_save_json(IntPtr gen, IntPtr disc, string path);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_train_load_json(IntPtr gen, IntPtr disc, string path);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_train_save_checkpoint(IntPtr gen, IntPtr disc, int ep, string dir);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_train_load_checkpoint(IntPtr gen, IntPtr disc, int ep, string dir);

    // ─── Metrics ──────────────────────────────────────────────────────────────
    [DllImport(Lib)] internal static extern void  gf_metrics_free(IntPtr m);
    [DllImport(Lib)] internal static extern float gf_metrics_d_loss_real(IntPtr m);
    [DllImport(Lib)] internal static extern float gf_metrics_d_loss_fake(IntPtr m);
    [DllImport(Lib)] internal static extern float gf_metrics_g_loss(IntPtr m);
    [DllImport(Lib)] internal static extern float gf_metrics_fid_score(IntPtr m);
    [DllImport(Lib)] internal static extern float gf_metrics_is_score(IntPtr m);
    [DllImport(Lib)] internal static extern float gf_metrics_grad_penalty(IntPtr m);
    [DllImport(Lib)] internal static extern int   gf_metrics_epoch(IntPtr m);
    [DllImport(Lib)] internal static extern int   gf_metrics_batch(IntPtr m);

    // ─── Result ───────────────────────────────────────────────────────────────
    [DllImport(Lib)] internal static extern IntPtr gf_run(IntPtr cfg);
    [DllImport(Lib)] internal static extern void   gf_result_free(IntPtr r);
    [DllImport(Lib)] internal static extern IntPtr gf_result_generator(IntPtr r);
    [DllImport(Lib)] internal static extern IntPtr gf_result_discriminator(IntPtr r);
    [DllImport(Lib)] internal static extern IntPtr gf_result_metrics(IntPtr r);

    // ─── Activations ──────────────────────────────────────────────────────────
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern IntPtr gf_activate(IntPtr a, string actType);
    [DllImport(Lib)] internal static extern IntPtr gf_relu(IntPtr a);
    [DllImport(Lib)] internal static extern IntPtr gf_sigmoid(IntPtr a);
    [DllImport(Lib)] internal static extern IntPtr gf_tanh_m(IntPtr a);
    [DllImport(Lib)] internal static extern IntPtr gf_leaky_relu(IntPtr a, float alpha);
    [DllImport(Lib)] internal static extern IntPtr gf_softmax(IntPtr a);

    // ─── Loss ─────────────────────────────────────────────────────────────────
    [DllImport(Lib)] internal static extern float  gf_bce_loss(IntPtr pred, IntPtr target);
    [DllImport(Lib)] internal static extern IntPtr gf_bce_grad(IntPtr pred, IntPtr target);
    [DllImport(Lib)] internal static extern float  gf_wgan_disc_loss(IntPtr dReal, IntPtr dFake);
    [DllImport(Lib)] internal static extern float  gf_wgan_gen_loss(IntPtr dFake);
    [DllImport(Lib)] internal static extern float  gf_hinge_disc_loss(IntPtr dReal, IntPtr dFake);
    [DllImport(Lib)] internal static extern float  gf_hinge_gen_loss(IntPtr dFake);
    [DllImport(Lib)] internal static extern float  gf_ls_disc_loss(IntPtr dReal, IntPtr dFake);
    [DllImport(Lib)] internal static extern float  gf_ls_gen_loss(IntPtr dFake);
    [DllImport(Lib)] internal static extern float  gf_cosine_anneal(int epoch, int maxEp, float baseLr, float minLr);

    // ─── Random ───────────────────────────────────────────────────────────────
    [DllImport(Lib)] internal static extern float  gf_random_gaussian();
    [DllImport(Lib)] internal static extern float  gf_random_uniform(float lo, float hi);
    [DllImport(Lib, CharSet = CharSet.Ansi)]
    internal static extern IntPtr gf_generate_noise(int size, int depth, string noiseType);

    // ─── Security ─────────────────────────────────────────────────────────────
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern int  gf_validate_path(string path);
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void gf_audit_log(string msg, string logFile);

    // ─── Backend ──────────────────────────────────────────────────────────────
    [DllImport(Lib, CharSet = CharSet.Ansi)] internal static extern void   gf_init_backend(string name);
    [DllImport(Lib)]                         internal static extern IntPtr gf_detect_backend();  // static string
    [DllImport(Lib)]                         internal static extern void   gf_secure_randomize();
}

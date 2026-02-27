// MIT License  Copyright (c) 2025 Matthew Abbott

namespace FacadedGan;

/// <summary>
/// GAN training hyper-parameters. Construct with <c>new Config()</c>.
/// Boolean fields are surfaced as <c>bool</c> properties.
/// Enum-typed fields accept lowercase strings (e.g. "relu", "adam", "bce").
/// String path/key fields are write-only (the C API has no getters for them).
/// </summary>
public sealed class Config : IDisposable
{
    private IntPtr _ptr;
    private bool _disposed;

    /// <summary>Create a Config pre-filled with library defaults.</summary>
    public Config()
    {
        _ptr = Native.gf_config_create();
        if (_ptr == IntPtr.Zero) throw new InvalidOperationException("gf_config_create failed.");
    }

    ~Config() => Dispose(false);
    /// <inheritdoc/>
    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    private void Dispose(bool _)
    {
        if (!_disposed && _ptr != IntPtr.Zero)
        {
            Native.gf_config_free(_ptr);
            _ptr = IntPtr.Zero;
            _disposed = true;
        }
    }

    internal IntPtr Handle => _disposed
        ? throw new ObjectDisposedException(nameof(Config))
        : _ptr;

    // ── Integer properties ────────────────────────────────────────────────────

    public int Epochs             { get => Native.gf_config_get_epochs(_ptr);             set => Native.gf_config_set_epochs(_ptr, value); }
    public int BatchSize          { get => Native.gf_config_get_batch_size(_ptr);         set => Native.gf_config_set_batch_size(_ptr, value); }
    public int NoiseDepth         { get => Native.gf_config_get_noise_depth(_ptr);        set => Native.gf_config_set_noise_depth(_ptr, value); }
    public int ConditionSize      { get => Native.gf_config_get_condition_size(_ptr);     set => Native.gf_config_set_condition_size(_ptr, value); }
    public int GeneratorBits      { get => Native.gf_config_get_generator_bits(_ptr);     set => Native.gf_config_set_generator_bits(_ptr, value); }
    public int DiscriminatorBits  { get => Native.gf_config_get_discriminator_bits(_ptr); set => Native.gf_config_set_discriminator_bits(_ptr, value); }
    public int MaxResLevel        { get => Native.gf_config_get_max_res_level(_ptr);      set => Native.gf_config_set_max_res_level(_ptr, value); }
    public int MetricInterval     { get => Native.gf_config_get_metric_interval(_ptr);    set => Native.gf_config_set_metric_interval(_ptr, value); }
    public int CheckpointInterval { get => Native.gf_config_get_checkpoint_interval(_ptr);set => Native.gf_config_set_checkpoint_interval(_ptr, value); }
    public int FuzzIterations     { get => Native.gf_config_get_fuzz_iterations(_ptr);    set => Native.gf_config_set_fuzz_iterations(_ptr, value); }
    public int NumThreads         { get => Native.gf_config_get_num_threads(_ptr);        set => Native.gf_config_set_num_threads(_ptr, value); }

    // ── Float properties ──────────────────────────────────────────────────────

    public float LearningRate   { get => Native.gf_config_get_learning_rate(_ptr);   set => Native.gf_config_set_learning_rate(_ptr, value); }
    public float GpLambda       { get => Native.gf_config_get_gp_lambda(_ptr);       set => Native.gf_config_set_gp_lambda(_ptr, value); }
    public float GeneratorLr    { get => Native.gf_config_get_generator_lr(_ptr);    set => Native.gf_config_set_generator_lr(_ptr, value); }
    public float DiscriminatorLr{ get => Native.gf_config_get_discriminator_lr(_ptr);set => Native.gf_config_set_discriminator_lr(_ptr, value); }
    public float WeightDecayVal { get => Native.gf_config_get_weight_decay_val(_ptr);set => Native.gf_config_set_weight_decay_val(_ptr, value); }

    // ── Bool properties ───────────────────────────────────────────────────────

    private static bool B(int v) => v != 0;
    private static int  I(bool v) => v ? 1 : 0;

    public bool UseBatchNorm      { get => B(Native.gf_config_get_use_batch_norm(_ptr));      set => Native.gf_config_set_use_batch_norm(_ptr, I(value)); }
    public bool UseLayerNorm      { get => B(Native.gf_config_get_use_layer_norm(_ptr));      set => Native.gf_config_set_use_layer_norm(_ptr, I(value)); }
    public bool UseSpectralNorm   { get => B(Native.gf_config_get_use_spectral_norm(_ptr));   set => Native.gf_config_set_use_spectral_norm(_ptr, I(value)); }
    public bool UseLabelSmoothing { get => B(Native.gf_config_get_use_label_smoothing(_ptr)); set => Native.gf_config_set_use_label_smoothing(_ptr, I(value)); }
    public bool UseFeatureMatching{ get => B(Native.gf_config_get_use_feature_matching(_ptr));set => Native.gf_config_set_use_feature_matching(_ptr, I(value)); }
    public bool UseMinibatchStdDev{ get => B(Native.gf_config_get_use_minibatch_std_dev(_ptr));set => Native.gf_config_set_use_minibatch_std_dev(_ptr, I(value)); }
    public bool UseProgressive    { get => B(Native.gf_config_get_use_progressive(_ptr));     set => Native.gf_config_set_use_progressive(_ptr, I(value)); }
    public bool UseAugmentation   { get => B(Native.gf_config_get_use_augmentation(_ptr));    set => Native.gf_config_set_use_augmentation(_ptr, I(value)); }
    public bool ComputeMetrics    { get => B(Native.gf_config_get_compute_metrics(_ptr));     set => Native.gf_config_set_compute_metrics(_ptr, I(value)); }
    public bool UseWeightDecay    { get => B(Native.gf_config_get_use_weight_decay(_ptr));    set => Native.gf_config_set_use_weight_decay(_ptr, I(value)); }
    public bool UseCosineAnneal   { get => B(Native.gf_config_get_use_cosine_anneal(_ptr));   set => Native.gf_config_set_use_cosine_anneal(_ptr, I(value)); }
    public bool AuditLog          { get => B(Native.gf_config_get_audit_log(_ptr));           set => Native.gf_config_set_audit_log(_ptr, I(value)); }
    public bool UseEncryption     { get => B(Native.gf_config_get_use_encryption(_ptr));      set => Native.gf_config_set_use_encryption(_ptr, I(value)); }
    public bool UseConv           { get => B(Native.gf_config_get_use_conv(_ptr));            set => Native.gf_config_set_use_conv(_ptr, I(value)); }
    public bool UseAttention      { get => B(Native.gf_config_get_use_attention(_ptr));       set => Native.gf_config_set_use_attention(_ptr, I(value)); }

    // ── String setters (C API provides no getters for string fields) ──────────

    /// <summary>Path to save the trained model.</summary>
    public string SaveModel      { set => Native.gf_config_set_save_model(_ptr, value); }
    /// <summary>Path to load a pretrained binary model.</summary>
    public string LoadModel      { set => Native.gf_config_set_load_model(_ptr, value); }
    /// <summary>Path to load a pretrained JSON model.</summary>
    public string LoadJsonModel  { set => Native.gf_config_set_load_json_model(_ptr, value); }
    /// <summary>Directory for generated outputs.</summary>
    public string OutputDir      { set => Native.gf_config_set_output_dir(_ptr, value); }
    /// <summary>Path to training data.</summary>
    public string DataPath       { set => Native.gf_config_set_data_path(_ptr, value); }
    /// <summary>Audit log file path.</summary>
    public string AuditLogFile   { set => Native.gf_config_set_audit_log_file(_ptr, value); }
    /// <summary>Encryption key for model files.</summary>
    public string EncryptionKey  { set => Native.gf_config_set_encryption_key(_ptr, value); }
    /// <summary>Patch configuration string.</summary>
    public string PatchConfig    { set => Native.gf_config_set_patch_config(_ptr, value); }

    // ── Enum setters (as strings) ─────────────────────────────────────────────

    /// <summary>Activation: "relu" | "sigmoid" | "tanh" | "leaky" | "none".</summary>
    public string Activation { set => Native.gf_config_set_activation(_ptr, value); }
    /// <summary>Noise type: "gauss" | "uniform" | "analog".</summary>
    public string NoiseType  { set => Native.gf_config_set_noise_type(_ptr, value); }
    /// <summary>Optimizer: "adam" | "sgd" | "rmsprop".</summary>
    public string Optimizer  { set => Native.gf_config_set_optimizer(_ptr, value); }
    /// <summary>Loss type: "bce" | "wgan" | "hinge" | "ls".</summary>
    public string LossType   { set => Native.gf_config_set_loss_type(_ptr, value); }
    /// <summary>Data type: "vector" | "image" | "audio".</summary>
    public string DataType   { set => Native.gf_config_set_data_type(_ptr, value); }
}

// MIT License  Copyright (c) 2025 Matthew Abbott
//
// Top-level static API for the facaded_gan GAN library.
//
// PREREQUISITES
// -------------
// Build the native library first:
//   cargo build --release -p facaded_gan_c
//
// Then ensure libfacaded_gan_c.so (Linux) / .dylib (macOS) / .dll (Windows)
// is in the library search path, or copy it next to your .NET assembly.
//
// EXAMPLE
// -------
//   using FacadedGan;
//
//   using var cfg = new Config { Epochs = 2, BatchSize = 8 };
//   Gan.InitBackend("cpu");
//   using var result = Gan.Run(cfg);
//   using var m = result.Metrics;
//   Console.WriteLine($"g_loss: {m.GLoss}");
//   using var gen = result.Generator;
//   Console.WriteLine($"gen layers: {gen.LayerCount}");

using System;
using System.Runtime.InteropServices;

namespace FacadedGan;

/// <summary>Top-level GAN API: high-level orchestration, training, loss, random, and security.</summary>
public static class Gan
{
    // ── High-level ────────────────────────────────────────────────────────────

    /// <summary>Build networks, train, and return results in one call.
    /// Caller owns the returned <see cref="GanResult"/>.</summary>
    public static GanResult Run(Config cfg) => new(Native.gf_run(cfg.Handle));

    /// <summary>Initialise the global compute backend.
    /// <paramref name="name"/>: "cpu" | "cuda" | "opencl" | "hybrid" | "auto".</summary>
    public static void InitBackend(string name) => Native.gf_init_backend(name);

    /// <summary>Detect the best available backend. Returns "cpu", "cuda", "opencl", or "hybrid".</summary>
    public static string DetectBackend()
    {
        var ptr = Native.gf_detect_backend();  // static string — do NOT free
        return Marshal.PtrToStringAnsi(ptr) ?? "cpu";
    }

    /// <summary>Seed the global RNG from /dev/urandom (or OS equivalent).</summary>
    public static void SecureRandomize() => Native.gf_secure_randomize();

    // ── Training ──────────────────────────────────────────────────────────────

    /// <summary>Run all epochs. Caller owns the returned <see cref="Metrics"/>.</summary>
    public static Metrics TrainFull(Network gen, Network disc, Dataset ds, Config cfg)
        => new(Native.gf_train_full(gen.Handle, disc.Handle, ds.Handle, cfg.Handle));

    /// <summary>Run one discriminator + generator update step.
    /// <paramref name="realBatch"/> is batch×features; <paramref name="noise"/> is batch×noiseDepth.
    /// Caller owns the returned <see cref="Metrics"/>.</summary>
    public static Metrics TrainStep(Network gen, Network disc,
                                    Matrix realBatch, Matrix noise, Config cfg)
        => new(Native.gf_train_step(gen.Handle, disc.Handle,
                                    realBatch.Handle, noise.Handle, cfg.Handle));

    /// <summary>Save both networks to a JSON file.</summary>
    public static void SaveJson(Network gen, Network disc, string path)
        => Native.gf_train_save_json(gen.Handle, disc.Handle, path);

    /// <summary>Load both networks from a JSON file.</summary>
    public static void LoadJson(Network gen, Network disc, string path)
        => Native.gf_train_load_json(gen.Handle, disc.Handle, path);

    /// <summary>Save a binary checkpoint to <paramref name="dir"/> at epoch <paramref name="epoch"/>.</summary>
    public static void SaveCheckpoint(Network gen, Network disc, int epoch, string dir)
        => Native.gf_train_save_checkpoint(gen.Handle, disc.Handle, epoch, dir);

    /// <summary>Load a binary checkpoint from <paramref name="dir"/> at epoch <paramref name="epoch"/>.</summary>
    public static void LoadCheckpoint(Network gen, Network disc, int epoch, string dir)
        => Native.gf_train_load_checkpoint(gen.Handle, disc.Handle, epoch, dir);

    // ── Loss ──────────────────────────────────────────────────────────────────

    public static float  BceLoss(Matrix pred, Matrix target)
        => Native.gf_bce_loss(pred.Handle, target.Handle);
    public static Matrix BceGrad(Matrix pred, Matrix target)
        => new(Native.gf_bce_grad(pred.Handle, target.Handle));
    public static float  WganDiscLoss(Matrix dReal, Matrix dFake)
        => Native.gf_wgan_disc_loss(dReal.Handle, dFake.Handle);
    public static float  WganGenLoss(Matrix dFake)
        => Native.gf_wgan_gen_loss(dFake.Handle);
    public static float  HingeDiscLoss(Matrix dReal, Matrix dFake)
        => Native.gf_hinge_disc_loss(dReal.Handle, dFake.Handle);
    public static float  HingeGenLoss(Matrix dFake)
        => Native.gf_hinge_gen_loss(dFake.Handle);
    public static float  LsDiscLoss(Matrix dReal, Matrix dFake)
        => Native.gf_ls_disc_loss(dReal.Handle, dFake.Handle);
    public static float  LsGenLoss(Matrix dFake)
        => Native.gf_ls_gen_loss(dFake.Handle);

    /// <summary>Cosine annealing LR schedule.</summary>
    public static float CosineAnneal(int epoch, int maxEp, float baseLr, float minLr)
        => Native.gf_cosine_anneal(epoch, maxEp, baseLr, minLr);

    // ── Random / Noise ────────────────────────────────────────────────────────

    public static float  RandomGaussian()               => Native.gf_random_gaussian();
    public static float  RandomUniform(float lo, float hi) => Native.gf_random_uniform(lo, hi);

    /// <summary>Generate a <paramref name="size"/>×<paramref name="depth"/> noise matrix.
    /// <paramref name="noiseType"/>: "gauss" | "uniform" | "analog".
    /// Caller owns the returned <see cref="Matrix"/>.</summary>
    public static Matrix GenerateNoise(int size, int depth, string noiseType)
        => new(Native.gf_generate_noise(size, depth, noiseType));

    // ── Security ──────────────────────────────────────────────────────────────

    /// <summary>Returns true if <paramref name="path"/> is safe (no traversal, etc.).</summary>
    public static bool ValidatePath(string path) => Native.gf_validate_path(path) != 0;

    /// <summary>Append <paramref name="msg"/> to <paramref name="logFile"/> with an ISO-8601 timestamp.</summary>
    public static void AuditLog(string msg, string logFile) => Native.gf_audit_log(msg, logFile);

    // ── Training extensions ───────────────────────────────────────────────────

    public static void   TrainOptimize(Network net) => Native.gf_train_optimize(net.Handle);
    public static void   TrainAdamUpdate(Matrix p, Matrix g, Matrix mBuf, Matrix vBuf, int t, float lr, float b1, float b2, float eps, float wd)
        => Native.gf_train_adam_update(p.Handle, g.Handle, mBuf.Handle, vBuf.Handle, t, lr, b1, b2, eps, wd);
    public static void   TrainSGDUpdate(Matrix p, Matrix g, float lr, float wd)
        => Native.gf_train_sgd_update(p.Handle, g.Handle, lr, wd);
    public static void   TrainRMSPropUpdate(Matrix p, Matrix g, Matrix cache, float lr, float decay, float eps, float wd)
        => Native.gf_train_rmsprop_update(p.Handle, g.Handle, cache.Handle, lr, decay, eps, wd);
    public static Matrix LabelSmoothing(Matrix labels, float lo, float hi)
        => new(Native.gf_train_label_smoothing(labels.Handle, lo, hi));
    public static Dataset LoadBMP(string path) => new(Native.gf_train_load_bmp(path));
    public static Dataset LoadWAV(string path) => new(Native.gf_train_load_wav(path));
    public static Matrix  Augment(Matrix sample, string dataType)
        => new(Native.gf_train_augment(sample.Handle, dataType));
    public static void LogMetrics(Metrics m, string filename)
        => Native.gf_train_log_metrics(m.Handle, filename);
    public static void SaveSamples(Network gen, int ep, string dir, int noiseDim, string noiseType)
        => Native.gf_train_save_samples(gen.Handle, ep, dir, noiseDim, noiseType);
    public static void PlotCSV(string filename, float[] dLoss, float[] gLoss)
        => Native.gf_train_plot_csv(filename, dLoss, gLoss, Math.Min(dLoss.Length, gLoss.Length));
    public static void PrintBar(float dLoss, float gLoss, int width = 40)
        => Native.gf_train_print_bar(dLoss, gLoss, width);
    public static float ComputeFID(MatrixArray realArr, MatrixArray fakeArr)
        => Native.gf_train_compute_fid(realArr.Handle, fakeArr.Handle);
    public static float ComputeIS(MatrixArray samples)
        => Native.gf_train_compute_is(samples.Handle);

    // ── Security extensions ───────────────────────────────────────────────────

    public static byte SecGetOSRandom() => Native.gf_sec_get_os_random();
    public static void SecEncryptModel(string inF, string outF, string key)
        => Native.gf_sec_encrypt_model(inF, outF, key);
    public static void SecDecryptModel(string inF, string outF, string key)
        => Native.gf_sec_decrypt_model(inF, outF, key);
    public static int SecRunTests()                    => Native.gf_sec_run_tests();
    public static int SecRunFuzzTests(int iterations)  => Native.gf_sec_run_fuzz_tests(iterations);

    // ── Disc standalone op ────────────────────────────────────────────────────

    public static Matrix DiscMinibatchStdDev(Matrix inp)
        => new(Native.gf_disc_minibatch_std_dev(inp.Handle));
}

// ── GanLayer ──────────────────────────────────────────────────────────────────

/// <summary>
/// A single neural-network layer (dense, conv, norm, or attention).
/// Created via static factory methods; dispose when done.
/// </summary>
public sealed class GanLayer : IDisposable
{
    private IntPtr _ptr;
    private bool _disposed;

    internal GanLayer(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) throw new InvalidOperationException("gf_layer_create_* returned null.");
        _ptr = ptr;
    }

    ~GanLayer() => Dispose(false);
    /// <inheritdoc/>
    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    private void Dispose(bool _)
    {
        if (!_disposed && _ptr != IntPtr.Zero)
        {
            Native.gf_layer_free(_ptr);
            _ptr = IntPtr.Zero;
            _disposed = true;
        }
    }

    internal IntPtr Handle => _disposed
        ? throw new ObjectDisposedException(nameof(GanLayer))
        : _ptr;

    // ── Factories ─────────────────────────────────────────────────────────────
    public static GanLayer Dense(int inSz, int outSz, string act = "relu")
        => new(Native.gf_layer_create_dense(inSz, outSz, act));
    public static GanLayer Conv2D(int inCh, int outCh, int kSz, int stride, int pad, int w, int h, string act = "leaky")
        => new(Native.gf_layer_create_conv2d(inCh, outCh, kSz, stride, pad, w, h, act));
    public static GanLayer Deconv2D(int inCh, int outCh, int kSz, int stride, int pad, int w, int h, string act = "relu")
        => new(Native.gf_layer_create_deconv2d(inCh, outCh, kSz, stride, pad, w, h, act));
    public static GanLayer Conv1D(int inCh, int outCh, int kSz, int stride, int pad, int inLen, string act = "leaky")
        => new(Native.gf_layer_create_conv1d(inCh, outCh, kSz, stride, pad, inLen, act));
    public static GanLayer BatchNorm(int features)  => new(Native.gf_layer_create_batch_norm(features));
    public static GanLayer LayerNorm(int features)  => new(Native.gf_layer_create_layer_norm(features));
    public static GanLayer Attention(int dModel, int nHeads) => new(Native.gf_layer_create_attention(dModel, nHeads));

    // ── Operations ────────────────────────────────────────────────────────────
    public Matrix Forward(Matrix inp)   => new(Native.gf_layer_forward(Handle, inp.Handle));
    public Matrix Backward(Matrix grad) => new(Native.gf_layer_backward(Handle, grad.Handle));
    public void   InitOptimizer(string opt = "adam") => Native.gf_layer_init_optimizer(Handle, opt);
    public Matrix Conv2DFwd(Matrix inp)         => new(Native.gf_layer_conv2d(inp.Handle, Handle));
    public Matrix Conv2DBwd(Matrix grad)        => new(Native.gf_layer_conv2d_backward(Handle, grad.Handle));
    public Matrix Deconv2DFwd(Matrix inp)       => new(Native.gf_layer_deconv2d(inp.Handle, Handle));
    public Matrix Deconv2DBwd(Matrix grad)      => new(Native.gf_layer_deconv2d_backward(Handle, grad.Handle));
    public Matrix Conv1DFwd(Matrix inp)         => new(Native.gf_layer_conv1d(inp.Handle, Handle));
    public Matrix Conv1DBwd(Matrix grad)        => new(Native.gf_layer_conv1d_backward(Handle, grad.Handle));
    public Matrix BatchNormFwd(Matrix inp)      => new(Native.gf_layer_batch_norm(inp.Handle, Handle));
    public Matrix BatchNormBwd(Matrix grad)     => new(Native.gf_layer_batch_norm_backward(Handle, grad.Handle));
    public Matrix LayerNormFwd(Matrix inp)      => new(Native.gf_layer_layer_norm(inp.Handle, Handle));
    public Matrix LayerNormBwd(Matrix grad)     => new(Native.gf_layer_layer_norm_backward(Handle, grad.Handle));
    public Matrix SpectralNorm()                => new(Native.gf_layer_spectral_norm(Handle));
    public Matrix AttentionFwd(Matrix inp)      => new(Native.gf_layer_attention(inp.Handle, Handle));
    public Matrix AttentionBwd(Matrix grad)     => new(Native.gf_layer_attention_backward(Handle, grad.Handle));
    public void   VerifyWeights()               => Native.gf_layer_verify_weights(Handle);
}

// ── MatrixArray ───────────────────────────────────────────────────────────────

/// <summary>A growable array of <see cref="Matrix"/> values used for FID / IS metrics.</summary>
public sealed class MatrixArray : IDisposable
{
    private IntPtr _ptr;
    private bool _disposed;

    /// <summary>Create an empty array.</summary>
    public MatrixArray() : this(Native.gf_matrix_array_create()) { }

    internal MatrixArray(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) throw new InvalidOperationException("gf_matrix_array_create returned null.");
        _ptr = ptr;
    }

    ~MatrixArray() => Dispose(false);
    /// <inheritdoc/>
    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    private void Dispose(bool _)
    {
        if (!_disposed && _ptr != IntPtr.Zero)
        {
            Native.gf_matrix_array_free(_ptr);
            _ptr = IntPtr.Zero;
            _disposed = true;
        }
    }

    internal IntPtr Handle => _disposed
        ? throw new ObjectDisposedException(nameof(MatrixArray))
        : _ptr;

    /// <summary>Append a copy of <paramref name="m"/> to the array.</summary>
    public void Push(Matrix m) => Native.gf_matrix_array_push(Handle, m.Handle);
    /// <summary>Number of matrices in the array.</summary>
    public int  Len()          => Native.gf_matrix_array_len(Handle);
}

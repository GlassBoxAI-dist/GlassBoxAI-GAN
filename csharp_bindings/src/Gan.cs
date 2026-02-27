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
}

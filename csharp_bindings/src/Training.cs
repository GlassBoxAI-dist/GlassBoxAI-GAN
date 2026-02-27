// MIT License  Copyright (c) 2025 Matthew Abbott

namespace FacadedGan;

/// <summary>A training dataset. Obtain via <see cref="Synthetic"/> or <see cref="Load"/>.</summary>
public sealed class Dataset : IDisposable
{
    private IntPtr _ptr;
    private bool _disposed;

    internal Dataset(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) throw new InvalidOperationException("gf_dataset_* returned null.");
        _ptr = ptr;
    }

    ~Dataset() => Dispose(false);
    /// <inheritdoc/>
    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    private void Dispose(bool _)
    {
        if (!_disposed && _ptr != IntPtr.Zero)
        {
            Native.gf_dataset_free(_ptr);
            _ptr = IntPtr.Zero;
            _disposed = true;
        }
    }

    internal IntPtr Handle => _disposed
        ? throw new ObjectDisposedException(nameof(Dataset))
        : _ptr;

    /// <summary>Create a synthetic random dataset with <paramref name="count"/> samples,
    /// each of length <paramref name="features"/>.</summary>
    public static Dataset Synthetic(int count, int features)
        => new(Native.gf_dataset_create_synthetic(count, features));

    /// <summary>Load a dataset from <paramref name="path"/>.
    /// <paramref name="dataType"/>: "vector" | "image" | "audio".</summary>
    public static Dataset Load(string path, string dataType)
        => new(Native.gf_dataset_load(path, dataType));

    /// <summary>Number of samples.</summary>
    public int Count => Native.gf_dataset_count(Handle);

    public override string ToString() => $"Dataset(count={Count})";
}

/// <summary>Per-step training statistics.</summary>
public sealed class Metrics : IDisposable
{
    private IntPtr _ptr;
    private bool _disposed;

    internal Metrics(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) throw new InvalidOperationException("gf_metrics returned null.");
        _ptr = ptr;
    }

    ~Metrics() => Dispose(false);
    /// <inheritdoc/>
    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    private void Dispose(bool _)
    {
        if (!_disposed && _ptr != IntPtr.Zero)
        {
            Native.gf_metrics_free(_ptr);
            _ptr = IntPtr.Zero;
            _disposed = true;
        }
    }

    internal IntPtr Handle => _disposed
        ? throw new ObjectDisposedException(nameof(Metrics))
        : _ptr;

    public float DLossReal   => Native.gf_metrics_d_loss_real(Handle);
    public float DLossFake   => Native.gf_metrics_d_loss_fake(Handle);
    public float GLoss       => Native.gf_metrics_g_loss(Handle);
    public float FidScore    => Native.gf_metrics_fid_score(Handle);
    public float IsScore     => Native.gf_metrics_is_score(Handle);
    public float GradPenalty => Native.gf_metrics_grad_penalty(Handle);
    public int   Epoch       => Native.gf_metrics_epoch(Handle);
    public int   Batch       => Native.gf_metrics_batch(Handle);

    public override string ToString()
        => $"Metrics(d={DLossReal:F4}/{DLossFake:F4}, g={GLoss:F4}, ep={Epoch}, batch={Batch})";
}

/// <summary>
/// Combined result of <see cref="Gan.Run"/>: trained networks and final metrics.
/// Each property allocates a new owned object; dispose each one when done.
/// </summary>
public sealed class GanResult : IDisposable
{
    private IntPtr _ptr;
    private bool _disposed;

    internal GanResult(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) throw new InvalidOperationException("gf_run returned null.");
        _ptr = ptr;
    }

    ~GanResult() => Dispose(false);
    /// <inheritdoc/>
    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    private void Dispose(bool _)
    {
        if (!_disposed && _ptr != IntPtr.Zero)
        {
            Native.gf_result_free(_ptr);
            _ptr = IntPtr.Zero;
            _disposed = true;
        }
    }

    internal IntPtr Handle => _disposed
        ? throw new ObjectDisposedException(nameof(GanResult))
        : _ptr;

    /// <summary>Cloned trained generator. Caller owns the returned <see cref="Network"/>.</summary>
    public Network Generator     => new(Native.gf_result_generator(Handle));

    /// <summary>Cloned trained discriminator. Caller owns the returned <see cref="Network"/>.</summary>
    public Network Discriminator => new(Native.gf_result_discriminator(Handle));

    /// <summary>Final training metrics. Caller owns the returned <see cref="Metrics"/>.</summary>
    public Metrics Metrics       => new(Native.gf_result_metrics(Handle));
}

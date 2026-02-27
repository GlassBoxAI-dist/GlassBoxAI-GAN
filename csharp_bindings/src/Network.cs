// MIT License  Copyright (c) 2025 Matthew Abbott

namespace FacadedGan;

/// <summary>
/// A generator or discriminator network.
/// Obtain via the static factory methods <see cref="GenBuild"/>, <see cref="DiscBuild"/>, etc.,
/// or from <see cref="GanResult.Generator"/> / <see cref="GanResult.Discriminator"/>.
/// </summary>
public sealed class Network : IDisposable
{
    private IntPtr _ptr;
    private bool _disposed;

    internal Network(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) throw new InvalidOperationException("gf_*_build returned null.");
        _ptr = ptr;
    }

    ~Network() => Dispose(false);
    /// <inheritdoc/>
    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    private void Dispose(bool _)
    {
        if (!_disposed && _ptr != IntPtr.Zero)
        {
            Native.gf_network_free(_ptr);
            _ptr = IntPtr.Zero;
            _disposed = true;
        }
    }

    internal IntPtr Handle => _disposed
        ? throw new ObjectDisposedException(nameof(Network))
        : _ptr;

    // ── Properties ────────────────────────────────────────────────────────────

    /// <summary>Number of layers.</summary>
    public int   LayerCount   => Native.gf_network_layer_count(Handle);
    /// <summary>Current learning rate.</summary>
    public float LearningRate => Native.gf_network_learning_rate(Handle);
    /// <summary>True if in training mode.</summary>
    public bool  IsTraining   => Native.gf_network_is_training(Handle) != 0;

    // ── Factory methods ───────────────────────────────────────────────────────

    /// <summary>Build a dense generator.
    /// <paramref name="sizes"/> lists layer widths, e.g. <c>[64, 128, 1]</c>.</summary>
    public static Network GenBuild(int[] sizes, string act, string opt, float lr)
        => new(Native.gf_gen_build(sizes, sizes.Length, act, opt, lr));

    /// <summary>Build a convolutional generator.</summary>
    public static Network GenBuildConv(int noiseDim, int condSz, int baseCh,
                                       string act, string opt, float lr)
        => new(Native.gf_gen_build_conv(noiseDim, condSz, baseCh, act, opt, lr));

    /// <summary>Build a dense discriminator.</summary>
    public static Network DiscBuild(int[] sizes, string act, string opt, float lr)
        => new(Native.gf_disc_build(sizes, sizes.Length, act, opt, lr));

    /// <summary>Build a convolutional discriminator.</summary>
    public static Network DiscBuildConv(int inCh, int inW, int inH,
                                        int condSz, int baseCh,
                                        string act, string opt, float lr)
        => new(Native.gf_disc_build_conv(inCh, inW, inH, condSz, baseCh, act, opt, lr));

    // ── Methods ───────────────────────────────────────────────────────────────

    /// <summary>Run a forward pass. <paramref name="inp"/> is batch×features.
    /// Caller owns the returned <see cref="Matrix"/>.</summary>
    public Matrix Forward(Matrix inp)
        => new(Native.gf_network_forward(Handle, inp.Handle));

    /// <summary>Run a backward pass. Caller owns the returned <see cref="Matrix"/>.</summary>
    public Matrix Backward(Matrix gradOut)
        => new(Native.gf_network_backward(Handle, gradOut.Handle));

    /// <summary>Apply accumulated gradients to weights.</summary>
    public void UpdateWeights() => Native.gf_network_update_weights(Handle);

    /// <summary>Switch training (<c>true</c>) / inference (<c>false</c>) mode.</summary>
    public void SetTraining(bool training)
        => Native.gf_network_set_training(Handle, training ? 1 : 0);

    /// <summary>Generate <paramref name="count"/> samples.
    /// <paramref name="noiseType"/>: "gauss" | "uniform" | "analog".
    /// Caller owns the returned <see cref="Matrix"/>.</summary>
    public Matrix Sample(int count, int noiseDim, string noiseType)
        => new(Native.gf_network_sample(Handle, count, noiseDim, noiseType));

    /// <summary>Sanitise weights (replace NaN/Inf with 0).</summary>
    public void Verify() => Native.gf_network_verify(Handle);

    /// <summary>Save weights to <paramref name="path"/> (binary format).</summary>
    public void Save(string path) => Native.gf_network_save(Handle, path);

    /// <summary>Load weights from <paramref name="path"/> (binary format).</summary>
    public void Load(string path) => Native.gf_network_load(Handle, path);

    public override string ToString() => $"Network(layers={LayerCount}, lr={LearningRate:G4})";
}

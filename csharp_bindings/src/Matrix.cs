// MIT License  Copyright (c) 2025 Matthew Abbott

using System.Runtime.InteropServices;

namespace FacadedGan;

/// <summary>
/// A 2-D row-major matrix of <c>float</c> values.
/// Wraps an opaque <c>GanMatrix*</c> from the C library.
/// Use in a <c>using</c> statement or call <see cref="Dispose"/> for deterministic release.
/// </summary>
public sealed class Matrix : IDisposable
{
    private IntPtr _ptr;
    private bool _disposed;

    internal Matrix(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) throw new InvalidOperationException("gf_matrix_* returned null.");
        _ptr = ptr;
    }

    /// <summary>Create a zero-filled <paramref name="rows"/>×<paramref name="cols"/> matrix.</summary>
    public Matrix(int rows, int cols) : this(Native.gf_matrix_create(rows, cols)) { }

    /// <summary>Create a Matrix by copying a flat row-major float array.
    /// <c>data[i*cols+j]</c> is row i, column j.</summary>
    public static Matrix FromArray(float[] data, int rows, int cols)
        => new(Native.gf_matrix_from_data(data, rows, cols));

    ~Matrix() => Dispose(false);

    /// <inheritdoc/>
    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }

    private void Dispose(bool _)
    {
        if (!_disposed && _ptr != IntPtr.Zero)
        {
            Native.gf_matrix_free(_ptr);
            _ptr = IntPtr.Zero;
            _disposed = true;
        }
    }

    internal IntPtr Handle => _disposed
        ? throw new ObjectDisposedException(nameof(Matrix))
        : _ptr;

    // ── Shape ─────────────────────────────────────────────────────────────────

    /// <summary>Number of rows.</summary>
    public int Rows => Native.gf_matrix_rows(Handle);

    /// <summary>Number of columns.</summary>
    public int Cols => Native.gf_matrix_cols(Handle);

    // ── Element access ────────────────────────────────────────────────────────

    /// <summary>Element read/write. Out-of-range reads return 0; writes are no-ops.</summary>
    public float this[int row, int col]
    {
        get => Native.gf_matrix_get(Handle, row, col);
        set => Native.gf_matrix_set(Handle, row, col, value);
    }

    /// <summary>Safe element read, returning <paramref name="def"/> if out-of-range.</summary>
    public float SafeGet(int row, int col, float def = 0f)
        => Native.gf_matrix_safe_get(Handle, row, col, def);

    /// <summary>Returns true if (r, c) is within bounds.</summary>
    public bool BoundsCheck(int r, int c) => Native.gf_bounds_check(Handle, r, c) != 0;

    // ── Data export ───────────────────────────────────────────────────────────

    /// <summary>Copy all elements into a flat row-major <c>float[]</c>.</summary>
    public float[] ToFlatArray()
    {
        int n = Rows * Cols;
        var data = new float[n];
        Marshal.Copy(Native.gf_matrix_data(Handle), data, 0, n);
        return data;
    }

    /// <summary>Copy all elements into a jagged <c>float[][]</c> (rows first).</summary>
    public float[][] ToArray()
    {
        int r = Rows, c = Cols;
        var flat = ToFlatArray();
        var out_ = new float[r][];
        for (int i = 0; i < r; i++)
        {
            out_[i] = new float[c];
            Array.Copy(flat, i * c, out_[i], 0, c);
        }
        return out_;
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// <summary>Matrix multiply A×B. Returns a new Matrix.</summary>
    public Matrix Multiply(Matrix b)   => new(Native.gf_matrix_multiply(Handle, b.Handle));

    /// <summary>Element-wise A+B. Returns a new Matrix.</summary>
    public Matrix Add(Matrix b)        => new(Native.gf_matrix_add(Handle, b.Handle));

    /// <summary>Element-wise A−B. Returns a new Matrix.</summary>
    public Matrix Subtract(Matrix b)   => new(Native.gf_matrix_subtract(Handle, b.Handle));

    /// <summary>Scalar multiply. Returns a new Matrix.</summary>
    public Matrix Scale(float s)       => new(Native.gf_matrix_scale(Handle, s));

    /// <summary>Transpose. Returns a new Matrix.</summary>
    public Matrix Transpose()          => new(Native.gf_matrix_transpose(Handle));

    /// <summary>L2-normalise each row. Returns a new Matrix.</summary>
    public Matrix Normalize()          => new(Native.gf_matrix_normalize(Handle));

    /// <summary>Element-wise product A⊙B. Returns a new Matrix.</summary>
    public Matrix ElementMul(Matrix b) => new(Native.gf_matrix_element_mul(Handle, b.Handle));

    public static Matrix operator *(Matrix a, Matrix b) => a.Multiply(b);
    public static Matrix operator +(Matrix a, Matrix b) => a.Add(b);
    public static Matrix operator -(Matrix a, Matrix b) => a.Subtract(b);
    public static Matrix operator *(Matrix a, float s)  => a.Scale(s);
    public static Matrix operator *(float s, Matrix a)  => a.Scale(s);

    // ── Activations ───────────────────────────────────────────────────────────

    /// <summary>Apply ReLU. Returns a new Matrix.</summary>
    public Matrix ReLU()                 => new(Native.gf_relu(Handle));

    /// <summary>Apply sigmoid. Returns a new Matrix.</summary>
    public Matrix Sigmoid()              => new(Native.gf_sigmoid(Handle));

    /// <summary>Apply tanh. Returns a new Matrix.</summary>
    public Matrix TanhAct()              => new(Native.gf_tanh_m(Handle));

    /// <summary>Apply leaky ReLU with <paramref name="alpha"/>. Returns a new Matrix.</summary>
    public Matrix LeakyReLU(float alpha) => new(Native.gf_leaky_relu(Handle, alpha));

    /// <summary>Apply softmax. Returns a new Matrix.</summary>
    public Matrix Softmax()              => new(Native.gf_softmax(Handle));

    /// <summary>Apply a named activation.
    /// act: "relu" | "sigmoid" | "tanh" | "leaky" | "none". Returns a new Matrix.</summary>
    public Matrix Activate(string act)   => new(Native.gf_activate(Handle, act));

    public override string ToString() => $"Matrix({Rows}×{Cols})";
}

/// <summary>
/// A 1-D float vector.
/// Wraps an opaque <c>GanVector*</c> from the C library.
/// </summary>
public sealed class Vector : IDisposable
{
    private IntPtr _ptr;
    private bool _disposed;

    internal Vector(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) throw new InvalidOperationException("gf_vector_* returned null.");
        _ptr = ptr;
    }

    /// <summary>Create a zero-filled vector of length <paramref name="length"/>.</summary>
    public Vector(int length) : this(Native.gf_vector_create(length)) { }

    ~Vector() => Dispose(false);
    /// <inheritdoc/>
    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    private void Dispose(bool _)
    {
        if (!_disposed && _ptr != IntPtr.Zero)
        {
            Native.gf_vector_free(_ptr);
            _ptr = IntPtr.Zero;
            _disposed = true;
        }
    }

    internal IntPtr Handle => _disposed
        ? throw new ObjectDisposedException(nameof(Vector))
        : _ptr;

    /// <summary>Number of elements.</summary>
    public int Length => Native.gf_vector_len(Handle);

    /// <summary>Element read (bounds-checked).</summary>
    public float this[int idx] => Native.gf_vector_get(Handle, idx);

    /// <summary>Copy elements into a <c>float[]</c>.</summary>
    public float[] ToArray()
    {
        int n = Length;
        var data = new float[n];
        Marshal.Copy(Native.gf_vector_data(Handle), data, 0, n);
        return data;
    }

    /// <summary>Spherical linear interpolation between this and <paramref name="other"/> at t ∈ [0,1].</summary>
    public Vector Slerp(Vector other, float t)
        => new(Native.gf_vector_noise_slerp(Handle, other.Handle, t));

    public override string ToString() => $"Vector(len={Length})";
}

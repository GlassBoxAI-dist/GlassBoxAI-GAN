"""
    FacadedGan

Julia bindings for the facaded_gan_cuda GAN library via `ccall`.

## Prerequisites

Build the C shared library first:

```sh
cargo build --release -p facaded_gan_c
```

## Quick start

```julia
using FacadedGan

init_backend("cpu")

cfg = Config()
cfg.epochs     = 2
cfg.batch_size = 8

result = run_gan(cfg)
println("g_loss: ", result.metrics.g_loss)
println("gen layers: ", result.generator.layer_count)

# Low-level
gen  = gen_build([64, 128, 1], "leaky", "adam", 0.0002f0)
out  = forward!(gen, Matrix(8, 64))
println("output: ", size(out))
```

## String enums

- activation : "relu" | "sigmoid" | "tanh" | "leaky" | "none"
- optimizer  : "adam" | "sgd" | "rmsprop"
- loss_type  : "bce"  | "wgan" | "hinge" | "ls"
- noise_type : "gauss"| "uniform" | "analog"
- data_type  : "vector"| "image" | "audio"
- backend    : "cpu"  | "cuda"  | "opencl" | "hybrid" | "auto"
"""
module FacadedGan

export Matrix, Vector, Config, Network, Dataset, Metrics, GanResult
export init_backend, detect_backend, secure_randomize, run_gan
export gen_build, gen_build_conv, disc_build, disc_build_conv
export train_full, train_step, save_json, load_json, save_checkpoint, load_checkpoint
export bce_loss, bce_grad, wgan_disc_loss, wgan_gen_loss
export hinge_disc_loss, hinge_gen_loss, ls_disc_loss, ls_gen_loss, cosine_anneal
export random_gaussian, random_uniform, generate_noise
export validate_path, audit_log
export free!

import Base: size, getindex, setindex!, *, +, -, show

# ─── Library path ─────────────────────────────────────────────────────────────

function _find_lib()
    base = joinpath(@__DIR__, "..", "..", "target")
    for subdir in ("release", "debug")
        for ext in (".so", ".dylib", ".dll")
            candidate = joinpath(base, subdir, string("libfacaded_gan_c", ext))
            isfile(candidate) && return abspath(candidate)
        end
        # Windows without lib prefix
        candidate = joinpath(base, subdir, "facaded_gan_c.dll")
        isfile(candidate) && return abspath(candidate)
    end
    error("""
    libfacaded_gan_c not found under $(abspath(base)).
    Build it first:
        cargo build --release -p facaded_gan_c
    """)
end

const _lib = _find_lib()

# ─── Internal helpers ─────────────────────────────────────────────────────────

_check(ptr::Ptr{Cvoid}, what) = ptr == C_NULL && error("$what returned NULL")

# Attach a free finalizer and return the object.
function _finalize!(obj, free_sym::Symbol)
    finalizer(obj) do x
        if x.ptr != C_NULL
            ccall((free_sym, _lib), Cvoid, (Ptr{Cvoid},), x.ptr)
            x.ptr = C_NULL
        end
    end
    obj
end

# ─── Matrix ───────────────────────────────────────────────────────────────────

"""
    Matrix(rows, cols)

Create a zero-filled `rows × cols` matrix backed by a C `GanMatrix`.
Memory is released automatically by the GC finalizer, or explicitly with `free!`.
"""
mutable struct Matrix
    ptr::Ptr{Cvoid}

    function Matrix(rows::Integer, cols::Integer)
        ptr = ccall((:gf_matrix_create, _lib), Ptr{Cvoid}, (Cint, Cint), rows, cols)
        _check(ptr, "gf_matrix_create")
        _finalize!(new(ptr), :gf_matrix_free)
    end

    # Internal: adopt a raw pointer returned from a C call.
    Matrix(ptr::Ptr{Cvoid}, ::Val{:adopt}) = _finalize!(new(ptr), :gf_matrix_free)
end

_mat(ptr::Ptr{Cvoid}) = (_check(ptr, "gf_*"); Matrix(ptr, Val(:adopt)))

"""
    free!(m::Matrix)

Release the C memory immediately. After this, `m` must not be used.
"""
function free!(m::Matrix)
    if m.ptr != C_NULL
        ccall((:gf_matrix_free, _lib), Cvoid, (Ptr{Cvoid},), m.ptr)
        m.ptr = C_NULL
        GC.safepoint()
    end
end

"""Create a Matrix from a Julia `Matrix{Float32}` (column-major → row-major)."""
function Matrix(data::AbstractMatrix{Float32})
    rows, cols = size(data)
    flat = vec(permutedims(data))  # row-major
    ptr  = ccall((:gf_matrix_from_data, _lib), Ptr{Cvoid},
                 (Ptr{Cfloat}, Cint, Cint), flat, rows, cols)
    _mat(ptr)
end

# ── Shape ─────────────────────────────────────────────────────────────────────

rows(m::Matrix) = Int(ccall((:gf_matrix_rows, _lib), Cint, (Ptr{Cvoid},), m.ptr))
cols(m::Matrix) = Int(ccall((:gf_matrix_cols, _lib), Cint, (Ptr{Cvoid},), m.ptr))
Base.size(m::Matrix) = (rows(m), cols(m))
Base.size(m::Matrix, d::Int) = size(m)[d]

# ── Element access ────────────────────────────────────────────────────────────

"""Element read (1-based; returns 0f0 on out-of-range)."""
Base.getindex(m::Matrix, i::Int, j::Int) =
    Float32(ccall((:gf_matrix_get, _lib), Cfloat, (Ptr{Cvoid}, Cint, Cint),
                  m.ptr, i - 1, j - 1))

"""Element write (1-based; no-op on out-of-range)."""
Base.setindex!(m::Matrix, v::Real, i::Int, j::Int) =
    ccall((:gf_matrix_set, _lib), Cvoid, (Ptr{Cvoid}, Cint, Cint, Cfloat),
          m.ptr, i - 1, j - 1, Float32(v))

"""Safe element read (1-based); returns `def` on out-of-range."""
safe_get(m::Matrix, i::Int, j::Int, def::Real=0f0) =
    Float32(ccall((:gf_matrix_safe_get, _lib), Cfloat,
                  (Ptr{Cvoid}, Cint, Cint, Cfloat), m.ptr, i-1, j-1, Float32(def)))

"""Returns true if (i, j) (1-based) is within bounds."""
bounds_check(m::Matrix, i::Int, j::Int) =
    ccall((:gf_bounds_check, _lib), Cint, (Ptr{Cvoid}, Cint, Cint),
          m.ptr, i-1, j-1) != 0

# ── Data export ───────────────────────────────────────────────────────────────

"""Copy the matrix into a Julia `Matrix{Float32}` (row → column major)."""
function Base.Matrix(m::Matrix)
    r, c = size(m)
    ptr  = ccall((:gf_matrix_data, _lib), Ptr{Cfloat}, (Ptr{Cvoid},), m.ptr)
    flat = unsafe_wrap(Array, Ptr{Float32}(ptr), r * c; own=false)
    permutedims(reshape(copy(flat), c, r))
end

"""Copy the matrix into a flat row-major `Vector{Float32}`."""
function to_flat(m::Matrix)
    r, c = size(m)
    ptr  = ccall((:gf_matrix_data, _lib), Ptr{Cfloat}, (Ptr{Cvoid},), m.ptr)
    unsafe_wrap(Array, Ptr{Float32}(ptr), r * c; own=false) |> copy
end

# ── Arithmetic ────────────────────────────────────────────────────────────────

Base.:*(a::Matrix, b::Matrix) =
    _mat(ccall((:gf_matrix_multiply,   _lib), Ptr{Cvoid}, (Ptr{Cvoid},Ptr{Cvoid}), a.ptr, b.ptr))
Base.:+(a::Matrix, b::Matrix) =
    _mat(ccall((:gf_matrix_add,        _lib), Ptr{Cvoid}, (Ptr{Cvoid},Ptr{Cvoid}), a.ptr, b.ptr))
Base.:-(a::Matrix, b::Matrix) =
    _mat(ccall((:gf_matrix_subtract,   _lib), Ptr{Cvoid}, (Ptr{Cvoid},Ptr{Cvoid}), a.ptr, b.ptr))
Base.:*(a::Matrix, s::Real) =
    _mat(ccall((:gf_matrix_scale,      _lib), Ptr{Cvoid}, (Ptr{Cvoid},Cfloat), a.ptr, Float32(s)))
Base.:*(s::Real, a::Matrix) = a * s

transpose(a::Matrix)   = _mat(ccall((:gf_matrix_transpose,   _lib), Ptr{Cvoid}, (Ptr{Cvoid},), a.ptr))
normalize(a::Matrix)   = _mat(ccall((:gf_matrix_normalize,   _lib), Ptr{Cvoid}, (Ptr{Cvoid},), a.ptr))
element_mul(a::Matrix, b::Matrix) =
    _mat(ccall((:gf_matrix_element_mul, _lib), Ptr{Cvoid}, (Ptr{Cvoid},Ptr{Cvoid}), a.ptr, b.ptr))

# ── Activations ───────────────────────────────────────────────────────────────

relu(a::Matrix)                = _mat(ccall((:gf_relu,       _lib), Ptr{Cvoid}, (Ptr{Cvoid},),         a.ptr))
sigmoid(a::Matrix)             = _mat(ccall((:gf_sigmoid,    _lib), Ptr{Cvoid}, (Ptr{Cvoid},),         a.ptr))
tanh_act(a::Matrix)            = _mat(ccall((:gf_tanh_m,     _lib), Ptr{Cvoid}, (Ptr{Cvoid},),         a.ptr))
softmax(a::Matrix)             = _mat(ccall((:gf_softmax,    _lib), Ptr{Cvoid}, (Ptr{Cvoid},),         a.ptr))
leaky_relu(a::Matrix, α::Real) = _mat(ccall((:gf_leaky_relu, _lib), Ptr{Cvoid}, (Ptr{Cvoid},Cfloat),   a.ptr, Float32(α)))
activate(a::Matrix, act::AbstractString) =
    _mat(ccall((:gf_activate, _lib), Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), a.ptr, act))

Base.show(io::IO, m::Matrix) = print(io, "Matrix($(rows(m))×$(cols(m)))")

# ─── Vector ───────────────────────────────────────────────────────────────────

"""
    GanVector(len)

A 1-D float vector backed by a C `GanVector`.
Named `GanVector` to avoid collision with Julia's built-in `Vector`.
"""
mutable struct GanVector
    ptr::Ptr{Cvoid}

    function GanVector(len::Integer)
        ptr = ccall((:gf_vector_create, _lib), Ptr{Cvoid}, (Cint,), len)
        _check(ptr, "gf_vector_create")
        _finalize!(new(ptr), :gf_vector_free)
    end

    GanVector(ptr::Ptr{Cvoid}, ::Val{:adopt}) = _finalize!(new(ptr), :gf_vector_free)
end

_vec(ptr::Ptr{Cvoid}) = (_check(ptr, "gf_*"); GanVector(ptr, Val(:adopt)))

free!(v::GanVector) = (v.ptr != C_NULL && (ccall((:gf_vector_free, _lib), Cvoid, (Ptr{Cvoid},), v.ptr); v.ptr = C_NULL))

len(v::GanVector)       = Int(ccall((:gf_vector_len, _lib), Cint, (Ptr{Cvoid},), v.ptr))
Base.length(v::GanVector) = len(v)
Base.getindex(v::GanVector, i::Int) =
    Float32(ccall((:gf_vector_get, _lib), Cfloat, (Ptr{Cvoid}, Cint), v.ptr, i - 1))

"""Spherical linear interpolation at t ∈ [0,1]."""
slerp(v1::GanVector, v2::GanVector, t::Real) =
    _vec(ccall((:gf_vector_noise_slerp, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}, Cfloat), v1.ptr, v2.ptr, Float32(t)))

function Base.Vector{Float32}(v::GanVector)
    n   = len(v)
    ptr = ccall((:gf_vector_data, _lib), Ptr{Cfloat}, (Ptr{Cvoid},), v.ptr)
    copy(unsafe_wrap(Array, Ptr{Float32}(ptr), n; own=false))
end

Base.show(io::IO, v::GanVector) = print(io, "GanVector(len=$(len(v)))")

# ─── Config ───────────────────────────────────────────────────────────────────

"""
    Config()

GAN training hyper-parameters. Fields are accessed as properties:

```julia
cfg = Config()
cfg.epochs      = 100
cfg.batch_size  = 32
cfg.learning_rate = 0.0002f0
cfg.loss_type   = "wgan"
```
"""
mutable struct Config
    ptr::Ptr{Cvoid}

    function Config()
        ptr = ccall((:gf_config_create, _lib), Ptr{Cvoid}, ())
        _check(ptr, "gf_config_create")
        _finalize!(new(ptr), :gf_config_free)
    end
end

free!(c::Config) = (c.ptr != C_NULL && (ccall((:gf_config_free, _lib), Cvoid, (Ptr{Cvoid},), c.ptr); c.ptr = C_NULL))

# getproperty / setproperty! dispatch tables
const _CFG_INT_GET = Dict(
    :epochs               => :gf_config_get_epochs,
    :batch_size           => :gf_config_get_batch_size,
    :noise_depth          => :gf_config_get_noise_depth,
    :condition_size       => :gf_config_get_condition_size,
    :generator_bits       => :gf_config_get_generator_bits,
    :discriminator_bits   => :gf_config_get_discriminator_bits,
    :max_res_level        => :gf_config_get_max_res_level,
    :metric_interval      => :gf_config_get_metric_interval,
    :checkpoint_interval  => :gf_config_get_checkpoint_interval,
    :fuzz_iterations      => :gf_config_get_fuzz_iterations,
    :num_threads          => :gf_config_get_num_threads,
)
const _CFG_INT_SET = Dict(
    :epochs               => :gf_config_set_epochs,
    :batch_size           => :gf_config_set_batch_size,
    :noise_depth          => :gf_config_set_noise_depth,
    :condition_size       => :gf_config_set_condition_size,
    :generator_bits       => :gf_config_set_generator_bits,
    :discriminator_bits   => :gf_config_set_discriminator_bits,
    :max_res_level        => :gf_config_set_max_res_level,
    :metric_interval      => :gf_config_set_metric_interval,
    :checkpoint_interval  => :gf_config_set_checkpoint_interval,
    :fuzz_iterations      => :gf_config_set_fuzz_iterations,
    :num_threads          => :gf_config_set_num_threads,
)
const _CFG_F32_GET = Dict(
    :learning_rate    => :gf_config_get_learning_rate,
    :gp_lambda        => :gf_config_get_gp_lambda,
    :generator_lr     => :gf_config_get_generator_lr,
    :discriminator_lr => :gf_config_get_discriminator_lr,
    :weight_decay_val => :gf_config_get_weight_decay_val,
)
const _CFG_F32_SET = Dict(
    :learning_rate    => :gf_config_set_learning_rate,
    :gp_lambda        => :gf_config_set_gp_lambda,
    :generator_lr     => :gf_config_set_generator_lr,
    :discriminator_lr => :gf_config_set_discriminator_lr,
    :weight_decay_val => :gf_config_set_weight_decay_val,
)
const _CFG_BOOL_GET = Dict(
    :use_batch_norm       => :gf_config_get_use_batch_norm,
    :use_layer_norm       => :gf_config_get_use_layer_norm,
    :use_spectral_norm    => :gf_config_get_use_spectral_norm,
    :use_label_smoothing  => :gf_config_get_use_label_smoothing,
    :use_feature_matching => :gf_config_get_use_feature_matching,
    :use_minibatch_std_dev=> :gf_config_get_use_minibatch_std_dev,
    :use_progressive      => :gf_config_get_use_progressive,
    :use_augmentation     => :gf_config_get_use_augmentation,
    :compute_metrics      => :gf_config_get_compute_metrics,
    :use_weight_decay     => :gf_config_get_use_weight_decay,
    :use_cosine_anneal    => :gf_config_get_use_cosine_anneal,
    :audit_log            => :gf_config_get_audit_log,
    :use_encryption       => :gf_config_get_use_encryption,
    :use_conv             => :gf_config_get_use_conv,
    :use_attention        => :gf_config_get_use_attention,
)
const _CFG_BOOL_SET = Dict(
    :use_batch_norm       => :gf_config_set_use_batch_norm,
    :use_layer_norm       => :gf_config_set_use_layer_norm,
    :use_spectral_norm    => :gf_config_set_use_spectral_norm,
    :use_label_smoothing  => :gf_config_set_use_label_smoothing,
    :use_feature_matching => :gf_config_set_use_feature_matching,
    :use_minibatch_std_dev=> :gf_config_set_use_minibatch_std_dev,
    :use_progressive      => :gf_config_set_use_progressive,
    :use_augmentation     => :gf_config_set_use_augmentation,
    :compute_metrics      => :gf_config_set_compute_metrics,
    :use_weight_decay     => :gf_config_set_use_weight_decay,
    :use_cosine_anneal    => :gf_config_set_use_cosine_anneal,
    :audit_log            => :gf_config_set_audit_log,
    :use_encryption       => :gf_config_set_use_encryption,
    :use_conv             => :gf_config_set_use_conv,
    :use_attention        => :gf_config_set_use_attention,
)
# String-only setters (C API has no getters for these)
const _CFG_STR_SET = Dict(
    :save_model       => :gf_config_set_save_model,
    :load_model       => :gf_config_set_load_model,
    :load_json_model  => :gf_config_set_load_json_model,
    :output_dir       => :gf_config_set_output_dir,
    :data_path        => :gf_config_set_data_path,
    :audit_log_file   => :gf_config_set_audit_log_file,
    :encryption_key   => :gf_config_set_encryption_key,
    :patch_config     => :gf_config_set_patch_config,
    :activation       => :gf_config_set_activation,
    :noise_type       => :gf_config_set_noise_type,
    :optimizer        => :gf_config_set_optimizer,
    :loss_type        => :gf_config_set_loss_type,
    :data_type        => :gf_config_set_data_type,
)

function Base.getproperty(c::Config, name::Symbol)
    name === :ptr && return getfield(c, :ptr)
    if haskey(_CFG_INT_GET, name)
        sym = _CFG_INT_GET[name]
        return Int(ccall((sym, _lib), Cint, (Ptr{Cvoid},), c.ptr))
    end
    if haskey(_CFG_F32_GET, name)
        sym = _CFG_F32_GET[name]
        return Float32(ccall((sym, _lib), Cfloat, (Ptr{Cvoid},), c.ptr))
    end
    if haskey(_CFG_BOOL_GET, name)
        sym = _CFG_BOOL_GET[name]
        return ccall((sym, _lib), Cint, (Ptr{Cvoid},), c.ptr) != 0
    end
    throw(ArgumentError("Config has no readable field :$name"))
end

function Base.setproperty!(c::Config, name::Symbol, val)
    name === :ptr && (setfield!(c, :ptr, val); return)
    if haskey(_CFG_INT_SET, name)
        sym = _CFG_INT_SET[name]
        ccall((sym, _lib), Cvoid, (Ptr{Cvoid}, Cint), c.ptr, Cint(val)); return
    end
    if haskey(_CFG_F32_SET, name)
        sym = _CFG_F32_SET[name]
        ccall((sym, _lib), Cvoid, (Ptr{Cvoid}, Cfloat), c.ptr, Cfloat(val)); return
    end
    if haskey(_CFG_BOOL_SET, name)
        sym = _CFG_BOOL_SET[name]
        ccall((sym, _lib), Cvoid, (Ptr{Cvoid}, Cint), c.ptr, val ? Cint(1) : Cint(0)); return
    end
    if haskey(_CFG_STR_SET, name)
        sym = _CFG_STR_SET[name]
        ccall((sym, _lib), Cvoid, (Ptr{Cvoid}, Cstring), c.ptr, String(val)); return
    end
    throw(ArgumentError("Config has no settable field :$name"))
end

Base.show(io::IO, c::Config) =
    print(io, "Config(epochs=$(c.epochs), batch_size=$(c.batch_size), lr=$(c.learning_rate))")

# ─── Network ──────────────────────────────────────────────────────────────────

"""
    Network  (opaque)

A generator or discriminator network. Obtain via `gen_build`, `disc_build`,
`gen_build_conv`, `disc_build_conv`, or from `GanResult`.
"""
mutable struct Network
    ptr::Ptr{Cvoid}
    Network(ptr::Ptr{Cvoid}) = _finalize!(new(ptr), :gf_network_free)
end

_net(ptr::Ptr{Cvoid}) = (_check(ptr, "gf_*_build"); Network(ptr))
free!(n::Network) = (n.ptr != C_NULL && (ccall((:gf_network_free, _lib), Cvoid, (Ptr{Cvoid},), n.ptr); n.ptr = C_NULL))

layer_count(n::Network)   = Int(ccall((:gf_network_layer_count,   _lib), Cint,   (Ptr{Cvoid},), n.ptr))
learning_rate(n::Network) = Float32(ccall((:gf_network_learning_rate, _lib), Cfloat, (Ptr{Cvoid},), n.ptr))
is_training(n::Network)   = ccall((:gf_network_is_training, _lib), Cint, (Ptr{Cvoid},), n.ptr) != 0

"""Run a forward pass. `inp` is a batch×features `Matrix`."""
forward!(n::Network, inp::Matrix) =
    _mat(ccall((:gf_network_forward, _lib), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), n.ptr, inp.ptr))

"""Run a backward pass."""
backward!(n::Network, grad::Matrix) =
    _mat(ccall((:gf_network_backward, _lib), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), n.ptr, grad.ptr))

"""Apply accumulated gradients to weights."""
update_weights!(n::Network) =
    ccall((:gf_network_update_weights, _lib), Cvoid, (Ptr{Cvoid},), n.ptr)

"""Switch training (`true`) / inference (`false`) mode."""
set_training!(n::Network, training::Bool) =
    ccall((:gf_network_set_training, _lib), Cvoid, (Ptr{Cvoid}, Cint), n.ptr, training ? 1 : 0)

"""Generate `count` samples. `noise_type`: "gauss" | "uniform" | "analog"."""
sample(n::Network, count::Integer, noise_dim::Integer, noise_type::AbstractString="gauss") =
    _mat(ccall((:gf_network_sample, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Cint, Cint, Cstring), n.ptr, count, noise_dim, noise_type))

"""Sanitise weights (replace NaN/Inf with 0)."""
verify!(n::Network) = ccall((:gf_network_verify, _lib), Cvoid, (Ptr{Cvoid},), n.ptr)

"""Save weights to `path` (binary format)."""
save_network(n::Network, path::AbstractString) =
    ccall((:gf_network_save, _lib), Cvoid, (Ptr{Cvoid}, Cstring), n.ptr, path)

"""Load weights from `path` (binary format)."""
load_network!(n::Network, path::AbstractString) =
    ccall((:gf_network_load, _lib), Cvoid, (Ptr{Cvoid}, Cstring), n.ptr, path)

Base.show(io::IO, n::Network) =
    print(io, "Network(layers=$(layer_count(n)), lr=$(learning_rate(n)))")

# ─── Network factory functions ────────────────────────────────────────────────

"""Build a dense generator. `sizes` is a vector of layer widths, e.g. `[64, 128, 1]`."""
function gen_build(sizes::AbstractVector{<:Integer},
                   act::AbstractString="leaky",
                   opt::AbstractString="adam",
                   lr::Real=0.0002f0)
    csizes = Cint.(sizes)
    _net(ccall((:gf_gen_build, _lib), Ptr{Cvoid},
               (Ptr{Cint}, Cint, Cstring, Cstring, Cfloat),
               csizes, length(csizes), act, opt, Float32(lr)))
end

"""Build a convolutional generator."""
function gen_build_conv(noise_dim::Integer, cond_sz::Integer, base_ch::Integer,
                        act::AbstractString="leaky",
                        opt::AbstractString="adam",
                        lr::Real=0.0002f0)
    _net(ccall((:gf_gen_build_conv, _lib), Ptr{Cvoid},
               (Cint, Cint, Cint, Cstring, Cstring, Cfloat),
               noise_dim, cond_sz, base_ch, act, opt, Float32(lr)))
end

"""Build a dense discriminator."""
function disc_build(sizes::AbstractVector{<:Integer},
                    act::AbstractString="leaky",
                    opt::AbstractString="adam",
                    lr::Real=0.0002f0)
    csizes = Cint.(sizes)
    _net(ccall((:gf_disc_build, _lib), Ptr{Cvoid},
               (Ptr{Cint}, Cint, Cstring, Cstring, Cfloat),
               csizes, length(csizes), act, opt, Float32(lr)))
end

"""Build a convolutional discriminator."""
function disc_build_conv(in_ch::Integer, in_w::Integer, in_h::Integer,
                         cond_sz::Integer, base_ch::Integer,
                         act::AbstractString="leaky",
                         opt::AbstractString="adam",
                         lr::Real=0.0002f0)
    _net(ccall((:gf_disc_build_conv, _lib), Ptr{Cvoid},
               (Cint, Cint, Cint, Cint, Cint, Cstring, Cstring, Cfloat),
               in_ch, in_w, in_h, cond_sz, base_ch, act, opt, Float32(lr)))
end

# ─── Dataset ──────────────────────────────────────────────────────────────────

"""Training dataset. Obtain via `synthetic_dataset` or `load_dataset`."""
mutable struct Dataset
    ptr::Ptr{Cvoid}
    Dataset(ptr::Ptr{Cvoid}) = _finalize!(new(ptr), :gf_dataset_free)
end

_ds(ptr::Ptr{Cvoid}) = (_check(ptr, "gf_dataset_*"); Dataset(ptr))
free!(d::Dataset) = (d.ptr != C_NULL && (ccall((:gf_dataset_free, _lib), Cvoid, (Ptr{Cvoid},), d.ptr); d.ptr = C_NULL))

"""Create a synthetic random dataset."""
synthetic_dataset(count::Integer, features::Integer) =
    _ds(ccall((:gf_dataset_create_synthetic, _lib), Ptr{Cvoid}, (Cint, Cint), count, features))

"""Load a dataset from `path`. `data_type`: "vector" | "image" | "audio"."""
load_dataset(path::AbstractString, data_type::AbstractString="vector") =
    _ds(ccall((:gf_dataset_load, _lib), Ptr{Cvoid}, (Cstring, Cstring), path, data_type))

dataset_count(d::Dataset) = Int(ccall((:gf_dataset_count, _lib), Cint, (Ptr{Cvoid},), d.ptr))
Base.length(d::Dataset)   = dataset_count(d)
Base.show(io::IO, d::Dataset) = print(io, "Dataset(count=$(dataset_count(d)))")

# ─── Metrics ──────────────────────────────────────────────────────────────────

"""Per-step training statistics. Returned by `train_full`, `train_step`, and `GanResult.metrics`."""
mutable struct Metrics
    ptr::Ptr{Cvoid}
    Metrics(ptr::Ptr{Cvoid}) = _finalize!(new(ptr), :gf_metrics_free)
end

_met(ptr::Ptr{Cvoid}) = (_check(ptr, "gf_train_*"); Metrics(ptr))
free!(m::Metrics) = (m.ptr != C_NULL && (ccall((:gf_metrics_free, _lib), Cvoid, (Ptr{Cvoid},), m.ptr); m.ptr = C_NULL))

function Base.getproperty(m::Metrics, name::Symbol)
    name === :ptr         && return getfield(m, :ptr)
    name === :d_loss_real && return Float32(ccall((:gf_metrics_d_loss_real,  _lib), Cfloat, (Ptr{Cvoid},), m.ptr))
    name === :d_loss_fake && return Float32(ccall((:gf_metrics_d_loss_fake,  _lib), Cfloat, (Ptr{Cvoid},), m.ptr))
    name === :g_loss      && return Float32(ccall((:gf_metrics_g_loss,       _lib), Cfloat, (Ptr{Cvoid},), m.ptr))
    name === :fid_score   && return Float32(ccall((:gf_metrics_fid_score,    _lib), Cfloat, (Ptr{Cvoid},), m.ptr))
    name === :is_score    && return Float32(ccall((:gf_metrics_is_score,     _lib), Cfloat, (Ptr{Cvoid},), m.ptr))
    name === :grad_penalty&& return Float32(ccall((:gf_metrics_grad_penalty, _lib), Cfloat, (Ptr{Cvoid},), m.ptr))
    name === :epoch       && return Int(ccall((:gf_metrics_epoch, _lib), Cint, (Ptr{Cvoid},), m.ptr))
    name === :batch       && return Int(ccall((:gf_metrics_batch, _lib), Cint, (Ptr{Cvoid},), m.ptr))
    throw(ArgumentError("Metrics has no field :$name"))
end

Base.show(io::IO, m::Metrics) =
    print(io, "Metrics(d=$(m.d_loss_real)/$(m.d_loss_fake), g=$(m.g_loss), ep=$(m.epoch))")

# ─── GanResult ────────────────────────────────────────────────────────────────

"""
Combined result of `run_gan()`.
Access `result.generator`, `result.discriminator`, `result.metrics`.
Each returns a new owned object.
"""
mutable struct GanResult
    ptr::Ptr{Cvoid}
    GanResult(ptr::Ptr{Cvoid}) = _finalize!(new(ptr), :gf_result_free)
end

free!(r::GanResult) = (r.ptr != C_NULL && (ccall((:gf_result_free, _lib), Cvoid, (Ptr{Cvoid},), r.ptr); r.ptr = C_NULL))

function Base.getproperty(r::GanResult, name::Symbol)
    name === :ptr           && return getfield(r, :ptr)
    name === :generator     && return _net(ccall((:gf_result_generator,     _lib), Ptr{Cvoid}, (Ptr{Cvoid},), r.ptr))
    name === :discriminator && return _net(ccall((:gf_result_discriminator, _lib), Ptr{Cvoid}, (Ptr{Cvoid},), r.ptr))
    name === :metrics       && return _met(ccall((:gf_result_metrics,       _lib), Ptr{Cvoid}, (Ptr{Cvoid},), r.ptr))
    throw(ArgumentError("GanResult has no field :$name"))
end

Base.show(io::IO, r::GanResult) = print(io, "GanResult(ptr=$(r.ptr))")

# ─── High-level API ───────────────────────────────────────────────────────────

"""
    run_gan(cfg::Config) -> GanResult

Build networks, train, and return a `GanResult` — all from a single `Config`.
"""
run_gan(cfg::Config) = GanResult(
    let ptr = ccall((:gf_run, _lib), Ptr{Cvoid}, (Ptr{Cvoid},), cfg.ptr)
        _check(ptr, "gf_run"); ptr
    end
)

"""Initialise the global compute backend. `name`: "cpu" | "cuda" | "opencl" | "hybrid" | "auto"."""
init_backend(name::AbstractString) =
    ccall((:gf_init_backend, _lib), Cvoid, (Cstring,), name)

"""Detect the best available backend. Returns a string like "CPU", "CUDA", etc."""
detect_backend() = unsafe_string(ccall((:gf_detect_backend, _lib), Cstring, ()))

"""Seed the global RNG from /dev/urandom."""
secure_randomize() = ccall((:gf_secure_randomize, _lib), Cvoid, ())

# ─── Training API ─────────────────────────────────────────────────────────────

"""Run all epochs. Returns final `Metrics`."""
train_full(gen::Network, disc::Network, ds::Dataset, cfg::Config) =
    _met(ccall((:gf_train_full, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
               gen.ptr, disc.ptr, ds.ptr, cfg.ptr))

"""Run one update step. `real_batch` is batch×features; `noise` is batch×noise_depth."""
train_step(gen::Network, disc::Network,
           real_batch::Matrix, noise::Matrix, cfg::Config) =
    _met(ccall((:gf_train_step, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
               gen.ptr, disc.ptr, real_batch.ptr, noise.ptr, cfg.ptr))

save_json(gen::Network, disc::Network, path::AbstractString) =
    ccall((:gf_train_save_json, _lib), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), gen.ptr, disc.ptr, path)

load_json(gen::Network, disc::Network, path::AbstractString) =
    ccall((:gf_train_load_json, _lib), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), gen.ptr, disc.ptr, path)

save_checkpoint(gen::Network, disc::Network, ep::Integer, dir::AbstractString) =
    ccall((:gf_train_save_checkpoint, _lib), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Cstring),
          gen.ptr, disc.ptr, ep, dir)

load_checkpoint(gen::Network, disc::Network, ep::Integer, dir::AbstractString) =
    ccall((:gf_train_load_checkpoint, _lib), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Cstring),
          gen.ptr, disc.ptr, ep, dir)

# ─── Loss functions ───────────────────────────────────────────────────────────

bce_loss(pred::Matrix, target::Matrix) =
    Float32(ccall((:gf_bce_loss, _lib), Cfloat, (Ptr{Cvoid}, Ptr{Cvoid}), pred.ptr, target.ptr))
bce_grad(pred::Matrix, target::Matrix) =
    _mat(ccall((:gf_bce_grad, _lib), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), pred.ptr, target.ptr))
wgan_disc_loss(d_real::Matrix, d_fake::Matrix) =
    Float32(ccall((:gf_wgan_disc_loss, _lib), Cfloat, (Ptr{Cvoid}, Ptr{Cvoid}), d_real.ptr, d_fake.ptr))
wgan_gen_loss(d_fake::Matrix) =
    Float32(ccall((:gf_wgan_gen_loss, _lib), Cfloat, (Ptr{Cvoid},), d_fake.ptr))
hinge_disc_loss(d_real::Matrix, d_fake::Matrix) =
    Float32(ccall((:gf_hinge_disc_loss, _lib), Cfloat, (Ptr{Cvoid}, Ptr{Cvoid}), d_real.ptr, d_fake.ptr))
hinge_gen_loss(d_fake::Matrix) =
    Float32(ccall((:gf_hinge_gen_loss, _lib), Cfloat, (Ptr{Cvoid},), d_fake.ptr))
ls_disc_loss(d_real::Matrix, d_fake::Matrix) =
    Float32(ccall((:gf_ls_disc_loss, _lib), Cfloat, (Ptr{Cvoid}, Ptr{Cvoid}), d_real.ptr, d_fake.ptr))
ls_gen_loss(d_fake::Matrix) =
    Float32(ccall((:gf_ls_gen_loss, _lib), Cfloat, (Ptr{Cvoid},), d_fake.ptr))
cosine_anneal(epoch::Integer, max_ep::Integer, base_lr::Real, min_lr::Real) =
    Float32(ccall((:gf_cosine_anneal, _lib), Cfloat,
                  (Cint, Cint, Cfloat, Cfloat), epoch, max_ep, Float32(base_lr), Float32(min_lr)))

# ─── Random / Noise ───────────────────────────────────────────────────────────

random_gaussian() = Float32(ccall((:gf_random_gaussian, _lib), Cfloat, ()))
random_uniform(lo::Real, hi::Real) =
    Float32(ccall((:gf_random_uniform, _lib), Cfloat, (Cfloat, Cfloat), Float32(lo), Float32(hi)))

"""Generate a `size × depth` noise matrix. `noise_type`: "gauss" | "uniform" | "analog"."""
generate_noise(sz::Integer, depth::Integer, noise_type::AbstractString="gauss") =
    _mat(ccall((:gf_generate_noise, _lib), Ptr{Cvoid},
               (Cint, Cint, Cstring), sz, depth, noise_type))

# ─── Security ─────────────────────────────────────────────────────────────────

"""Returns `true` if `path` is safe (no directory traversal, etc.)."""
validate_path(path::AbstractString) =
    ccall((:gf_validate_path, _lib), Cint, (Cstring,), path) != 0

"""Append `msg` to `log_file` with an ISO-8601 timestamp."""
audit_log(msg::AbstractString, log_file::AbstractString) =
    ccall((:gf_audit_log, _lib), Cvoid, (Cstring, Cstring), msg, log_file)

# ─── Matrix in-place operations ───────────────────────────────────────────────

"""Add `b` into `a` in place."""
function matrix_add_in_place(a::Matrix, b::Matrix)
    ccall((:gf_matrix_add_in_place, _lib), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), a.ptr, b.ptr)
end

"""Scale `a` by `s` in place."""
function matrix_scale_in_place(a::Matrix, s::Real)
    ccall((:gf_matrix_scale_in_place, _lib), Cvoid, (Ptr{Cvoid}, Cfloat), a.ptr, Float32(s))
end

"""Clip all elements of `a` to [lo, hi] in place."""
function matrix_clip_in_place(a::Matrix, lo::Real, hi::Real)
    ccall((:gf_matrix_clip_in_place, _lib), Cvoid, (Ptr{Cvoid}, Cfloat, Cfloat),
          a.ptr, Float32(lo), Float32(hi))
end

"""Safe element write (1-based); no-op on out-of-range."""
function matrix_safe_set(m::Matrix, i::Int, j::Int, v::Real)
    ccall((:gf_matrix_safe_set, _lib), Cvoid, (Ptr{Cvoid}, Cint, Cint, Cfloat),
          m.ptr, i - 1, j - 1, Float32(v))
end

"""Backward pass through an activation function (element-wise derivative × `grad`)."""
function matrix_activation_backward(a::Matrix, grad::Matrix, act::AbstractString)
    _mat(ccall((:gf_matrix_activation_backward, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), a.ptr, grad.ptr, act))
end

# ─── GanLayer ─────────────────────────────────────────────────────────────────

"""
    GanLayer  (opaque)

A single network layer (dense, conv2d, deconv2d, conv1d, batch-norm,
layer-norm, or attention). Obtain via one of the `layer_create_*` functions.
Memory is released automatically by the GC finalizer, or explicitly with
`free!(layer)`.
"""
mutable struct GanLayer
    ptr::Ptr{Cvoid}

    GanLayer(ptr::Ptr{Cvoid}) = _finalize!(new(ptr), :gf_layer_free)
    GanLayer(ptr::Ptr{Cvoid}, ::Val{:adopt}) = _finalize!(new(ptr), :gf_layer_free)
end

_layer(ptr::Ptr{Cvoid}) = (_check(ptr, "gf_layer_*"); GanLayer(ptr, Val(:adopt)))

free!(l::GanLayer) = (l.ptr != C_NULL && (ccall((:gf_layer_free, _lib), Cvoid, (Ptr{Cvoid},), l.ptr); l.ptr = C_NULL))

Base.show(io::IO, l::GanLayer) = print(io, "GanLayer(ptr=$(l.ptr))")

# ── Layer factory functions ────────────────────────────────────────────────────

"""Create a dense (fully-connected) layer with `in_sz` inputs and `out_sz` outputs."""
function layer_create_dense(in_sz::Integer, out_sz::Integer,
                            act::AbstractString="relu",
                            opt::AbstractString="adam",
                            lr::Real=0.0002f0)
    _layer(ccall((:gf_layer_create_dense, _lib), Ptr{Cvoid},
                 (Cint, Cint, Cstring, Cstring, Cfloat),
                 in_sz, out_sz, act, opt, Float32(lr)))
end

"""Create a 2-D convolution layer."""
function layer_create_conv2d(in_ch::Integer, out_ch::Integer,
                             kernel::Integer, stride::Integer, pad::Integer,
                             act::AbstractString="relu",
                             opt::AbstractString="adam",
                             lr::Real=0.0002f0)
    _layer(ccall((:gf_layer_create_conv2d, _lib), Ptr{Cvoid},
                 (Cint, Cint, Cint, Cint, Cint, Cstring, Cstring, Cfloat),
                 in_ch, out_ch, kernel, stride, pad, act, opt, Float32(lr)))
end

"""Create a 2-D transposed convolution (deconv) layer."""
function layer_create_deconv2d(in_ch::Integer, out_ch::Integer,
                               kernel::Integer, stride::Integer, pad::Integer,
                               act::AbstractString="relu",
                               opt::AbstractString="adam",
                               lr::Real=0.0002f0)
    _layer(ccall((:gf_layer_create_deconv2d, _lib), Ptr{Cvoid},
                 (Cint, Cint, Cint, Cint, Cint, Cstring, Cstring, Cfloat),
                 in_ch, out_ch, kernel, stride, pad, act, opt, Float32(lr)))
end

"""Create a 1-D convolution layer."""
function layer_create_conv1d(in_ch::Integer, out_ch::Integer,
                             kernel::Integer, stride::Integer, pad::Integer,
                             act::AbstractString="relu",
                             opt::AbstractString="adam",
                             lr::Real=0.0002f0)
    _layer(ccall((:gf_layer_create_conv1d, _lib), Ptr{Cvoid},
                 (Cint, Cint, Cint, Cint, Cint, Cstring, Cstring, Cfloat),
                 in_ch, out_ch, kernel, stride, pad, act, opt, Float32(lr)))
end

"""Create a batch-normalisation layer with `features` feature channels."""
function layer_create_batch_norm(features::Integer, lr::Real=0.0002f0)
    _layer(ccall((:gf_layer_create_batch_norm, _lib), Ptr{Cvoid},
                 (Cint, Cfloat), features, Float32(lr)))
end

"""Create a layer-normalisation layer with `features` feature channels."""
function layer_create_layer_norm(features::Integer, lr::Real=0.0002f0)
    _layer(ccall((:gf_layer_create_layer_norm, _lib), Ptr{Cvoid},
                 (Cint, Cfloat), features, Float32(lr)))
end

"""Create a self-attention layer."""
function layer_create_attention(embed_dim::Integer, num_heads::Integer,
                                opt::AbstractString="adam",
                                lr::Real=0.0002f0)
    _layer(ccall((:gf_layer_create_attention, _lib), Ptr{Cvoid},
                 (Cint, Cint, Cstring, Cfloat),
                 embed_dim, num_heads, opt, Float32(lr)))
end

# ── Layer methods ──────────────────────────────────────────────────────────────

"""Run a forward pass through `l`. Returns output `Matrix`."""
layer_forward(l::GanLayer, inp::Matrix) =
    _mat(ccall((:gf_layer_forward, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), l.ptr, inp.ptr))

"""Run a backward pass through `l`. Returns gradient `Matrix`."""
layer_backward(l::GanLayer, grad::Matrix) =
    _mat(ccall((:gf_layer_backward, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), l.ptr, grad.ptr))

"""Initialise the optimizer state for `l`."""
layer_init_optimizer(l::GanLayer, opt::AbstractString, lr::Real) =
    ccall((:gf_layer_init_optimizer, _lib), Cvoid,
          (Ptr{Cvoid}, Cstring, Cfloat), l.ptr, opt, Float32(lr))

"""2-D convolution forward pass."""
layer_conv2d(l::GanLayer, inp::Matrix) =
    _mat(ccall((:gf_layer_conv2d, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), l.ptr, inp.ptr))

"""2-D convolution backward pass."""
layer_conv2d_backward(l::GanLayer, grad::Matrix) =
    _mat(ccall((:gf_layer_conv2d_backward, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), l.ptr, grad.ptr))

"""2-D transposed convolution forward pass."""
layer_deconv2d(l::GanLayer, inp::Matrix) =
    _mat(ccall((:gf_layer_deconv2d, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), l.ptr, inp.ptr))

"""2-D transposed convolution backward pass."""
layer_deconv2d_backward(l::GanLayer, grad::Matrix) =
    _mat(ccall((:gf_layer_deconv2d_backward, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), l.ptr, grad.ptr))

"""1-D convolution forward pass."""
layer_conv1d(l::GanLayer, inp::Matrix) =
    _mat(ccall((:gf_layer_conv1d, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), l.ptr, inp.ptr))

"""1-D convolution backward pass."""
layer_conv1d_backward(l::GanLayer, grad::Matrix) =
    _mat(ccall((:gf_layer_conv1d_backward, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), l.ptr, grad.ptr))

"""Batch normalisation forward pass."""
layer_batch_norm(l::GanLayer, inp::Matrix) =
    _mat(ccall((:gf_layer_batch_norm, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), l.ptr, inp.ptr))

"""Batch normalisation backward pass."""
layer_batch_norm_backward(l::GanLayer, grad::Matrix) =
    _mat(ccall((:gf_layer_batch_norm_backward, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), l.ptr, grad.ptr))

"""Layer normalisation forward pass."""
layer_layer_norm(l::GanLayer, inp::Matrix) =
    _mat(ccall((:gf_layer_layer_norm, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), l.ptr, inp.ptr))

"""Layer normalisation backward pass."""
layer_layer_norm_backward(l::GanLayer, grad::Matrix) =
    _mat(ccall((:gf_layer_layer_norm_backward, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), l.ptr, grad.ptr))

"""Apply spectral normalisation to `l`'s weights in place."""
layer_spectral_norm(l::GanLayer) =
    ccall((:gf_layer_spectral_norm, _lib), Cvoid, (Ptr{Cvoid},), l.ptr)

"""Self-attention forward pass."""
layer_attention(l::GanLayer, inp::Matrix) =
    _mat(ccall((:gf_layer_attention, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), l.ptr, inp.ptr))

"""Self-attention backward pass."""
layer_attention_backward(l::GanLayer, grad::Matrix) =
    _mat(ccall((:gf_layer_attention_backward, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), l.ptr, grad.ptr))

"""Sanitise weights in `l` (replace NaN/Inf with 0)."""
layer_verify_weights(l::GanLayer) =
    ccall((:gf_layer_verify_weights, _lib), Cvoid, (Ptr{Cvoid},), l.ptr)

# ─── GanMatrixArray ───────────────────────────────────────────────────────────

"""
    GanMatrixArray  (opaque)

A growable, heap-allocated array of `Matrix` objects managed on the C side.
Obtain via `matrix_array_create()` and release with `free!` or the GC finalizer.
"""
mutable struct GanMatrixArray
    ptr::Ptr{Cvoid}

    GanMatrixArray(ptr::Ptr{Cvoid}) = _finalize!(new(ptr), :gf_matrix_array_free)
    GanMatrixArray(ptr::Ptr{Cvoid}, ::Val{:adopt}) = _finalize!(new(ptr), :gf_matrix_array_free)
end

_marr(ptr::Ptr{Cvoid}) = (_check(ptr, "gf_matrix_array_*"); GanMatrixArray(ptr, Val(:adopt)))

free!(a::GanMatrixArray) = (a.ptr != C_NULL && (ccall((:gf_matrix_array_free, _lib), Cvoid, (Ptr{Cvoid},), a.ptr); a.ptr = C_NULL))

Base.show(io::IO, a::GanMatrixArray) = print(io, "GanMatrixArray(len=$(matrix_array_len(a)))")

"""Create an empty `GanMatrixArray`."""
matrix_array_create() =
    _marr(ccall((:gf_matrix_array_create, _lib), Ptr{Cvoid}, ()))

"""Append a `Matrix` to `arr` (the C side takes ownership of a reference)."""
matrix_array_push(arr::GanMatrixArray, m::Matrix) =
    ccall((:gf_matrix_array_push, _lib), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), arr.ptr, m.ptr)

"""Return the number of matrices stored in `arr`."""
matrix_array_len(arr::GanMatrixArray) =
    Int(ccall((:gf_matrix_array_len, _lib), Cint, (Ptr{Cvoid},), arr.ptr))

Base.length(arr::GanMatrixArray) = matrix_array_len(arr)

# ─── Network extensions ────────────────────────────────────────────────────────

"""
    gen_sample_conditional(gen, count, noise_dim, cond_sz, noise_type, cond) -> Matrix

Generate `count` conditional samples using `cond` (a `Matrix` of conditioning
vectors, one per row) and the given `noise_type`.
"""
gen_sample_conditional(gen::Network, count::Integer, noise_dim::Integer,
                       cond_sz::Integer, noise_type::AbstractString,
                       cond::Matrix) =
    _mat(ccall((:gf_gen_sample_conditional, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Cint, Cint, Cint, Cstring, Ptr{Cvoid}),
               gen.ptr, count, noise_dim, cond_sz, noise_type, cond.ptr))

"""Add a progressive-growth layer at resolution level `res_lvl` to a generator."""
gen_add_progressive_layer(gen::Network, res_lvl::Integer) =
    ccall((:gf_gen_add_progressive_layer, _lib), Cvoid, (Ptr{Cvoid}, Cint), gen.ptr, res_lvl)

"""Return the output `Matrix` of the layer at index `idx` (0-based) in a generator."""
gen_get_layer_output(gen::Network, idx::Integer) =
    _mat(ccall((:gf_gen_get_layer_output, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Cint), gen.ptr, idx))

"""Return a deep copy of generator `gen` as a new `Network`."""
gen_deep_copy(gen::Network) =
    _net(ccall((:gf_gen_deep_copy, _lib), Ptr{Cvoid}, (Ptr{Cvoid},), gen.ptr))

"""Run a discriminator forward pass on `inp`. Returns output `Matrix`."""
disc_evaluate(disc::Network, inp::Matrix) =
    _mat(ccall((:gf_disc_evaluate, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), disc.ptr, inp.ptr))

"""Compute the gradient penalty between `real` and `fake` with coefficient `lambda`."""
disc_grad_penalty(disc::Network, real::Matrix, fake::Matrix, lambda::Real) =
    Float32(ccall((:gf_disc_grad_penalty, _lib), Cfloat,
                  (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cfloat),
                  disc.ptr, real.ptr, fake.ptr, Float32(lambda)))

"""Compute the feature-matching loss between `real` and `fake` at layer `feat_layer`."""
disc_feature_match(disc::Network, real::Matrix, fake::Matrix, feat_layer::Integer) =
    Float32(ccall((:gf_disc_feature_match, _lib), Cfloat,
                  (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cint),
                  disc.ptr, real.ptr, fake.ptr, feat_layer))

"""Compute the minibatch standard deviation of `inp`. Returns a `Matrix`."""
disc_minibatch_std_dev(inp::Matrix) =
    _mat(ccall((:gf_disc_minibatch_std_dev, _lib), Ptr{Cvoid}, (Ptr{Cvoid},), inp.ptr))

"""Add a progressive-growth layer at resolution level `res_lvl` to a discriminator."""
disc_add_progressive_layer(disc::Network, res_lvl::Integer) =
    ccall((:gf_disc_add_progressive_layer, _lib), Cvoid, (Ptr{Cvoid}, Cint), disc.ptr, res_lvl)

"""Return the output `Matrix` of the layer at index `idx` (0-based) in a discriminator."""
disc_get_layer_output(disc::Network, idx::Integer) =
    _mat(ccall((:gf_disc_get_layer_output, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Cint), disc.ptr, idx))

"""Return a deep copy of discriminator `disc` as a new `Network`."""
disc_deep_copy(disc::Network) =
    _net(ccall((:gf_disc_deep_copy, _lib), Ptr{Cvoid}, (Ptr{Cvoid},), disc.ptr))

# ─── Training extensions ──────────────────────────────────────────────────────

"""Apply the optimizer update step to all trainable parameters in `net`."""
train_optimize(net::Network) =
    ccall((:gf_train_optimize, _lib), Cvoid, (Ptr{Cvoid},), net.ptr)

"""
    train_adam_update(p, g, m_buf, v_buf, t, lr, b1, b2, eps, wd)

In-place Adam parameter update.
`p`     – parameter matrix (updated in place)
`g`     – gradient matrix
`m_buf` – first-moment buffer (updated in place)
`v_buf` – second-moment buffer (updated in place)
`t`     – time step (integer, ≥ 1)
`lr`, `b1`, `b2`, `eps`, `wd` – Adam hyper-parameters
"""
train_adam_update(p::Matrix, g::Matrix, m_buf::Matrix, v_buf::Matrix,
                  t::Integer, lr::Real, b1::Real, b2::Real, eps::Real, wd::Real) =
    ccall((:gf_train_adam_update, _lib), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
           Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat),
          p.ptr, g.ptr, m_buf.ptr, v_buf.ptr,
          Cint(t), Float32(lr), Float32(b1), Float32(b2), Float32(eps), Float32(wd))

"""
    train_sgd_update(p, g, lr, wd)

In-place SGD (with optional weight decay) parameter update.
"""
train_sgd_update(p::Matrix, g::Matrix, lr::Real, wd::Real) =
    ccall((:gf_train_sgd_update, _lib), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}, Cfloat, Cfloat),
          p.ptr, g.ptr, Float32(lr), Float32(wd))

"""
    train_rmsprop_update(p, g, cache, lr, decay, eps, wd)

In-place RMSProp parameter update.
`cache` – running-mean-square buffer (updated in place).
"""
train_rmsprop_update(p::Matrix, g::Matrix, cache::Matrix,
                     lr::Real, decay::Real, eps::Real, wd::Real) =
    ccall((:gf_train_rmsprop_update, _lib), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cfloat, Cfloat, Cfloat, Cfloat),
          p.ptr, g.ptr, cache.ptr, Float32(lr), Float32(decay), Float32(eps), Float32(wd))

"""Apply label smoothing to `labels`, clamping values to [lo, hi]. Returns a new `Matrix`."""
train_label_smoothing(labels::Matrix, lo::Real, hi::Real) =
    _mat(ccall((:gf_train_label_smoothing, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Cfloat, Cfloat), labels.ptr, Float32(lo), Float32(hi)))

"""Load a BMP image file as a `Dataset`."""
train_load_bmp(path::AbstractString) =
    _ds(ccall((:gf_train_load_bmp, _lib), Ptr{Cvoid}, (Cstring,), path))

"""Load a WAV audio file as a `Dataset`."""
train_load_wav(path::AbstractString) =
    _ds(ccall((:gf_train_load_wav, _lib), Ptr{Cvoid}, (Cstring,), path))

"""Augment a single `sample` for the given `data_type`. Returns a new `Matrix`."""
train_augment(sample::Matrix, data_type::AbstractString) =
    _mat(ccall((:gf_train_augment, _lib), Ptr{Cvoid},
               (Ptr{Cvoid}, Cstring), sample.ptr, data_type))

"""Append a `Metrics` snapshot to `filename` (CSV/text log)."""
train_log_metrics(m::Metrics, filename::AbstractString) =
    ccall((:gf_train_log_metrics, _lib), Cvoid, (Ptr{Cvoid}, Cstring), m.ptr, filename)

"""Save generator samples for epoch `ep` into `dir`."""
train_save_samples(gen::Network, ep::Integer, dir::AbstractString,
                   noise_dim::Integer, noise_type::AbstractString) =
    ccall((:gf_train_save_samples, _lib), Cvoid,
          (Ptr{Cvoid}, Cint, Cstring, Cint, Cstring),
          gen.ptr, Cint(ep), dir, Cint(noise_dim), noise_type)

"""
    train_plot_csv(filename, d_loss, g_loss, cnt)

Write a CSV plot file with `cnt` (d_loss, g_loss) rows from the supplied
`Vector{Float32}` arrays.
"""
function train_plot_csv(filename::AbstractString,
                        d_loss::AbstractVector{Float32},
                        g_loss::AbstractVector{Float32},
                        cnt::Integer)
    ccall((:gf_train_plot_csv, _lib), Cvoid,
          (Cstring, Ptr{Cfloat}, Ptr{Cfloat}, Cint),
          filename, d_loss, g_loss, Cint(cnt))
end

"""Print a training progress bar to stdout."""
train_print_bar(d_loss::Real, g_loss::Real, width::Integer) =
    ccall((:gf_train_print_bar, _lib), Cvoid,
          (Cfloat, Cfloat, Cint), Float32(d_loss), Float32(g_loss), Cint(width))

"""Compute the Fréchet Inception Distance between `real_arr` and `fake_arr` sample sets."""
train_compute_fid(real_arr::GanMatrixArray, fake_arr::GanMatrixArray) =
    Float32(ccall((:gf_train_compute_fid, _lib), Cfloat,
                  (Ptr{Cvoid}, Ptr{Cvoid}), real_arr.ptr, fake_arr.ptr))

"""Compute the Inception Score for a `GanMatrixArray` of generated samples."""
train_compute_is(samples::GanMatrixArray) =
    Float32(ccall((:gf_train_compute_is, _lib), Cfloat, (Ptr{Cvoid},), samples.ptr))

# ─── Security extensions ──────────────────────────────────────────────────────

"""Return one random byte obtained from the OS entropy source."""
sec_get_os_random() =
    ccall((:gf_sec_get_os_random, _lib), UInt8, ())

"""Encrypt the model file at `in_f`, writing ciphertext to `out_f` using `key`."""
sec_encrypt_model(in_f::AbstractString, out_f::AbstractString, key::AbstractString) =
    ccall((:gf_sec_encrypt_model, _lib), Cvoid, (Cstring, Cstring, Cstring), in_f, out_f, key)

"""Decrypt the model file at `in_f`, writing plaintext to `out_f` using `key`."""
sec_decrypt_model(in_f::AbstractString, out_f::AbstractString, key::AbstractString) =
    ccall((:gf_sec_decrypt_model, _lib), Cvoid, (Cstring, Cstring, Cstring), in_f, out_f, key)

"""Run the built-in security test suite. Returns the number of failures (0 = all passed)."""
sec_run_tests() =
    Int(ccall((:gf_sec_run_tests, _lib), Cint, ()))

"""Run `iterations` rounds of fuzz testing. Returns the number of failures."""
sec_run_fuzz_tests(iterations::Integer) =
    Int(ccall((:gf_sec_run_fuzz_tests, _lib), Cint, (Cint,), Cint(iterations)))

end # module FacadedGan

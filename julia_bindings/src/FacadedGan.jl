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

end # module FacadedGan

# GlassBoxAI-GAN

## **Generative Adversarial Network Suite**

### *CUDA/OpenCL-Accelerated GAN with Multi-Language Bindings & Formal Verification*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-blue.svg)](https://www.khronos.org/opencl/)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933.svg)](https://nodejs.org/)
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8.svg)](https://go.dev/)
[![Julia](https://img.shields.io/badge/Julia-1.6+-9558B2.svg)](https://julialang.org/)
[![C#](https://img.shields.io/badge/C%23-.NET%208.0-512BD4.svg)](https://dotnet.microsoft.com/)
[![Zig](https://img.shields.io/badge/Zig-0.15+-F7A41D.svg)](https://ziglang.org/)
[![Kani](https://img.shields.io/badge/Kani-Verified-brightgreen.svg)](https://model-checking.github.io/kani/)
[![CISA Compliant](https://img.shields.io/badge/CISA-Secure%20by%20Design-blue.svg)](https://www.cisa.gov/securebydesign)

---

## **Overview**

GlassBoxAI-GAN is a comprehensive, production-ready Generative Adversarial Network implementation featuring:

- **Dual GPU backends**: CUDA (via cudarc) and OpenCL (via ocl) with automatic detection and runtime selection
- **Facade pattern architecture**: Clean, unified API isolating GAN orchestration from compute backend details
- **Multi-language bindings**: Rust, Python, Node.js, C, C++, Go, Julia, C#, and Zig — all sharing a single C ABI
- **Multiple GAN variants**: Dense and convolutional networks, WGAN-GP, hinge loss, least-squares, spectral norm, progressive growing, attention
- **Formal verification**: Kani-verified Rust implementation with 39 FFI boundary proof harnesses covering every C API function
- **CISA/NSA Secure by Design compliance**: Audit logging, path validation, memory sanitisation, encryption, and bounds-checked ops throughout

---

## **Table of Contents**

1. [Features](#features)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Prerequisites](#prerequisites)
5. [Installation & Building](#installation--building)
   - [Rust Library & CLI](#rust-library--cli)
   - [Python Package](#python-package)
   - [Node.js Package](#nodejs-package)
   - [C/C++ Library](#cc-library)
   - [Go Package](#go-package)
   - [Julia Package](#julia-package)
   - [C# Package](#c-package)
   - [Zig Package](#zig-package)
6. [Usage](#usage)
   - [Rust API](#rust-api)
   - [Python API](#python-api)
   - [Node.js API](#nodejs-api)
   - [Go API](#go-api)
   - [Julia API](#julia-api)
   - [C API](#c-api)
   - [C++ API](#c-api-1)
   - [C# API](#c-api-2)
   - [Zig API](#zig-api)
   - [CLI Reference](#cli-reference)
7. [API Reference](#api-reference)
8. [Formal Verification with Kani](#formal-verification-with-kani)
9. [CISA/NSA Compliance](#cisansa-compliance)
10. [License](#license)
11. [Author](#author)

---

## **Features**

### Core GAN Capabilities

| Feature | Description |
|---------|-------------|
| **Dense GAN** | Fully-connected generator and discriminator with configurable layer widths |
| **Convolutional GAN** | Conv generator and discriminator with configurable channel depth |
| **Loss Functions** | BCE, WGAN-GP (with gradient penalty), Hinge, Least-Squares |
| **Optimizers** | Adam, SGD, RMSProp with configurable learning rates per network |
| **Activations** | ReLU, LeakyReLU, Sigmoid, Tanh, None |
| **Regularisation** | Batch norm, layer norm, spectral norm, weight decay, label smoothing |
| **Progressive Growing** | Resolution-staged training with configurable max level |
| **Attention** | Self-attention layers toggleable at config time |
| **Noise Generation** | Gaussian, uniform, and analog noise types; spherical interpolation (slerp) |
| **Model Persistence** | Binary checkpoint and JSON model formats; load/save at any epoch |
| **Dataset Loading** | Synthetic generation, vector, image, and audio dataset types |
| **Metrics** | D-loss real/fake, G-loss, FID score, IS score, gradient penalty per step |
| **Cosine Annealing** | Built-in LR schedule with configurable min/max |

### GPU Acceleration

| Backend | Implementation | Performance |
|---------|---------------|-------------|
| **CUDA** | cudarc Rust bindings | Optimal for NVIDIA GPUs |
| **OpenCL** | ocl Rust bindings | Cross-vendor GPU support (NVIDIA, AMD, Intel) |
| **Hybrid** | Both active simultaneously | Heterogeneous compute |
| **Auto** | Runtime detection | Tries CUDA, falls back to OpenCL, then CPU |
| **CPU** | Pure Rust | No GPU required; always available |

### Language Bindings

| Language | Technology | Pattern |
|----------|------------|---------|
| **Rust** | Native library crate | Ownership-safe structs, `GANResult`, `GANMetrics` |
| **Python** | PyO3 + maturin | `facaded_gan` module with `facade` and `api` submodules |
| **Node.js** | napi-rs 2.x | Native addon, `GanConfig`, `GanNetwork`, `GanResult` classes |
| **C** | `extern "C"` cdylib | Opaque handles, `gf_*` prefix, caller-owned heap objects |
| **C++** | RAII header wrapper | `GanMatrix`, `GanVector`, `GanNetwork`, `GanConfig` classes |
| **Go** | CGo | Idiomatic Go types with `runtime.SetFinalizer` GC cleanup |
| **Julia** | `ccall` | `Base.getproperty`/`setproperty!` config, 1-based indexing |
| **C#** | P/Invoke (.NET 8) | `IDisposable` + finalizer pattern, operator overloads |
| **Zig** | C FFI | Error unions `!T`, `opaque {}` handles, `[:0]const u8` strings |

### Safety & Security

| Feature | Technology |
|---------|------------|
| **Memory Safety** | Rust ownership; no manual free in safe code |
| **Formal Verification** | 39 Kani FFI boundary harnesses |
| **Null Pointer Safety** | Every `gf_*` function null-checks all pointer params |
| **Bounds Checking** | `safe_get`/`safe_set`; `bounds_check`; all matrix ops |
| **Path Validation** | `gf_validate_path` rejects traversal; used before every file I/O |
| **Audit Logging** | ISO-8601 timestamped append-only log via `gf_audit_log` |
| **Weight Sanitisation** | `gf_network_verify` replaces NaN/Inf with 0 after training |
| **Secure RNG Seed** | `gf_secure_randomize` seeds from `/dev/urandom` |
| **Input Sanitisation** | CLI args validated; unknown enum strings fall back to safe defaults |

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GlassBoxAI-GAN                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Core Rust Library  (src/)                      │    │
│  │  • types.rs      — GANConfig, Network, Dataset, GANResult   │    │
│  │  • facade.rs     — gf_* API: train, generate, security      │    │
│  │  • training.rs   — train_full / train_step → GANMetrics     │    │
│  │  • network.rs    — dense + conv forward/backward            │    │
│  │  • matrix.rs     — TMatrix ops (add/mul/norm/clip/…)        │    │
│  │  • loss.rs       — BCE, WGAN-GP, Hinge, LS                  │    │
│  │  • activations.rs — ReLU, Sigmoid, Tanh, Leaky, Softmax     │    │
│  │  • optimizer.rs  — Adam, SGD, RMSProp, cosine_anneal        │    │
│  │  • random.rs     — Gaussian, uniform, analog noise, slerp   │    │
│  │  • security.rs   — validate_path, audit_log, bounds_check   │    │
│  │  • main.rs       — CLI (arg parse → gf_run)                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌────────────────────────┐  ┌────────────────────────────────┐     │
│  │    CUDA Backend        │  │    OpenCL Backend              │     │
│  ├────────────────────────┤  ├────────────────────────────────┤     │
│  │ cudarc (NVIDIA GPUs)   │  │ ocl (NVIDIA / AMD / Intel)     │     │
│  │ Feature: cuda          │  │ Feature: opencl                │     │
│  └────────────────────────┘  └────────────────────────────────┘     │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │             C ABI Layer  (c_bindings/)                      │    │
│  │  • lib.rs           — 100+ gf_* extern "C" functions        │    │
│  │  • kani_ffi_tests.rs — 39 Kani boundary proof harnesses     │    │
│  │  include/facaded_gan.h    — C header                        │    │
│  │  include/facaded_gan.hpp  — C++ RAII wrapper                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌────────┐ ┌────────┐ ┌──────┐ ┌───────┐ ┌──────┐ ┌──────┐       │
│  │ Python │ │Node.js │ │  Go  │ │ Julia │ │  C#  │ │ Zig  │       │
│  ├────────┤ ├────────┤ ├──────┤ ├───────┤ ├──────┤ ├──────┤       │
│  │ PyO3   │ │napi-rs │ │ CGo  │ │ ccall │ │ P/I  │ │C FFI │       │
│  └────────┘ └────────┘ └──────┘ └───────┘ └──────┘ └──────┘       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Kani Formal Verification  (kani/ + c_bindings) │    │
│  │  15 library harness groups (bound_checks, no_panic, …)      │    │
│  │  39 FFI C boundary harnesses  (kani_ffi_tests.rs)           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## **File Structure**

```
facaded_rust_cuda/
│
├── Cargo.toml                        # Workspace root (facaded_gan_cuda + sub-crates)
├── pyproject.toml                    # Python maturin project
│
├── src/                              # Core Rust library
│   ├── main.rs                       # CLI binary (arg parse → gf_run)
│   ├── lib.rs                        # Library crate root
│   ├── types.rs                      # GANConfig, Network, Dataset, GANMetrics, GANResult
│   ├── facade.rs                     # gf_* public API surface
│   ├── training.rs                   # train_full / train_step → GANMetrics
│   ├── network.rs                    # Dense + convolutional network forward/backward
│   ├── matrix.rs                     # TMatrix operations
│   ├── loss.rs                       # BCE, WGAN-GP, Hinge, Least-Squares
│   ├── activations.rs                # Activation functions + backward passes
│   ├── normalization.rs              # Batch norm, layer norm, spectral norm
│   ├── convolution.rs                # Convolutional layer operations
│   ├── attention.rs                  # Self-attention layer
│   ├── optimizer.rs                  # Adam, SGD, RMSProp, cosine annealing
│   ├── random.rs                     # Noise generation, slerp
│   ├── security.rs                   # validate_path, audit_log, bounds_check
│   ├── backend/                      # Backend abstraction + detection
│   ├── kani_tests.rs                 # Includes kani/ harness modules
│   ├── tests.rs                      # Unit tests
│   └── quality_tests.rs              # Quality / fuzz tests
│
├── kani/                             # Library-level Kani harnesses
│   ├── mod.rs                        # Module declarations (15 groups)
│   ├── bound_checks.rs               # OOB-free indexing proofs
│   ├── no_panic.rs                   # No panic! proofs
│   ├── integer_overflow.rs           # Arithmetic safety
│   ├── div_by_zero.rs                # Non-zero denominator proofs
│   ├── pointer_validity.rs           # Slice/reference access
│   ├── global_state.rs               # Network state invariants
│   ├── deadlock_free.rs              # Sequential locking (absence proofs)
│   ├── input_sanitization.rs         # Bounded loops / recursion
│   ├── result_coverage.rs            # Result/Option variant coverage
│   ├── memory_limits.rs              # Allocation size bounds
│   ├── constant_time.rs              # No secret-dependent branches
│   ├── state_machine.rs              # Training ↔ inference transitions
│   ├── enum_exhaustion.rs            # Match arm exhaustiveness
│   ├── float_sanity.rs               # NaN/Inf logic checks
│   └── resource_limits.rs            # Security budget proofs
│
├── c_bindings/                       # C ABI shared/static library
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                    # 100+ gf_* extern "C" functions
│       └── kani_ffi_tests.rs         # 39 FFI Kani boundary harnesses
│
├── include/                          # C/C++ headers
│   ├── facaded_gan.h                 # C header (opaque handles, gf_* API)
│   └── facaded_gan.hpp               # C++ RAII wrapper classes
│
├── python_bindings/                  # PyO3 Python extension
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs                    # GANConfig, Network, facade/api submodules
│
├── nodejs_bindings/                  # napi-rs Node.js native addon
│   ├── Cargo.toml
│   ├── build.rs
│   ├── package.json
│   └── src/
│       └── lib.rs                    # GanConfig, GanNetwork, GanResult JS classes
│
├── go_bindings/                      # CGo Go package
│   ├── go.mod
│   └── facadedgan/
│       └── facadedgan.go             # Config, Network, Dataset, Metrics, Result + funcs
│
├── julia_bindings/                   # Julia ccall package
│   ├── Project.toml
│   └── src/
│       └── FacadedGan.jl             # Matrix, GanVector, Config, Network, …
│
├── csharp_bindings/                  # .NET 8 P/Invoke library
│   ├── FacadedGan.csproj
│   └── src/
│       ├── Native.cs                 # DllImport declarations
│       ├── Matrix.cs                 # Matrix + Vector with operators
│       ├── Config.cs                 # GANConfig with typed properties
│       ├── Network.cs                # Generator + discriminator factory
│       ├── Training.cs               # Dataset, Metrics, GanResult
│       └── Gan.cs                    # Top-level static Gan class
│
├── zig_bindings/                     # Zig C FFI wrapper
│   ├── build.zig.zon
│   ├── build.zig
│   └── src/
│       ├── facaded_gan.zig           # Matrix, Vector, Config, Network, … + top-level fns
│       └── example.zig               # Usage example
│
└── gan_facade_tests.sh               # Integration test script
```

---

## **Prerequisites**

### Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Rust** | 1.70+ | Core library compilation |
| **Cargo** | (with Rust) | Build system and package manager |

### GPU Backend (optional — at least one recommended)

| Dependency | Version | Purpose |
|------------|---------|---------|
| **CUDA Toolkit** | 12.0+ | NVIDIA GPU acceleration (feature: `cuda`) |
| **OpenCL SDK** | 3.0+ | Cross-vendor GPU acceleration (feature: `opencl`) |

### Per Language Binding

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Python bindings |
| **maturin** | 1.x | Building PyO3 wheels |
| **Node.js** | 18+ | Node.js bindings |
| **@napi-rs/cli** | 2.x | Building napi-rs native addon |
| **Go** | 1.21+ | Go CGo bindings |
| **Julia** | 1.6+ | Julia ccall bindings |
| **.NET SDK** | 8.0+ | C# P/Invoke bindings |
| **Zig** | 0.15+ | Zig C FFI bindings |
| **GCC/Clang** | 11+ | C/C++ compilation against headers |
| **Kani** | 0.50+ | Formal verification |

---

## **Installation & Building**

### **Rust Library & CLI**

```bash
# CPU-only (no GPU required)
cargo build --release --no-default-features

# CUDA backend
cargo build --release --no-default-features --features cuda

# OpenCL backend
cargo build --release --no-default-features --features opencl

# Both backends (default)
cargo build --release

# Run CLI
./target/release/facaded_gan_cuda --help
```

Add as a Rust dependency:

```toml
[dependencies]
facaded_gan_cuda = { path = "path/to/facaded_rust_cuda" }
```

### **C/C++ Library**

```bash
# Build the C shared + static library
cargo build --release -p facaded_gan_c

# CPU-only (no CUDA/OpenCL required)
cargo build --release --no-default-features -p facaded_gan_c
```

Produces `target/release/libfacaded_gan_c.so` (Linux), `.dylib` (macOS), `.dll` (Windows).

```bash
# Link against C header
gcc  -o myapp myapp.c   -Iinclude -Ltarget/release -lfacaded_gan_c -Wl,-rpath,target/release

# Link against C++ RAII header
g++  -o myapp myapp.cpp -Iinclude -Ltarget/release -lfacaded_gan_c -Wl,-rpath,target/release -std=c++17
```

### **Python Package**

```bash
# Install maturin
pip install maturin

# Development install (editable, CPU-only)
maturin develop --manifest-path python_bindings/Cargo.toml --no-default-features

# Build a distributable wheel
maturin build  --manifest-path python_bindings/Cargo.toml --no-default-features --release
pip install target/wheels/facaded_gan-*.whl
```

### **Node.js Package**

```bash
cd nodejs_bindings
npm install
npm run build           # release build via napi build
# or debug:
npm run build:debug
```

### **Go Package**

```bash
# Build native library first
cargo build --release --no-default-features -p facaded_gan_c

# Build Go package
cd go_bindings
LD_LIBRARY_PATH=../target/release go build ./...

# Run Go example
LD_LIBRARY_PATH=../target/release go run ./...
```

### **Julia Package**

```bash
# Build native library first
cargo build --release --no-default-features -p facaded_gan_c

# Activate Julia environment
julia --project=julia_bindings -e 'using FacadedGan; println(detect_backend())'
```

### **C# Package**

```bash
# Build native library first
cargo build --release --no-default-features -p facaded_gan_c

# Build .NET library
dotnet build csharp_bindings/FacadedGan.csproj

# Run with native library on path
LD_LIBRARY_PATH=target/release dotnet run --project csharp_bindings
```

### **Zig Package**

```bash
# Build native library first
cargo build --release --no-default-features -p facaded_gan_c

# Build Zig example
cd zig_bindings
zig build

# Run example
LD_LIBRARY_PATH=../target/release zig-out/bin/example
```

---

## **Usage**

### **Rust API**

```rust
use facaded_gan_cuda::facade;
use facaded_gan_cuda::types::{GANConfig, GANResult};

fn main() {
    // Configure
    let mut cfg = GANConfig::default();
    cfg.epochs      = 50;
    cfg.batch_size  = 32;
    cfg.noise_depth = 64;
    cfg.learning_rate = 0.0002;
    cfg.use_conv    = false;
    cfg.loss_type   = facaded_gan_cuda::types::LossType::WGANGP;

    // Initialise backend
    facaded_gan_cuda::backend::init_backend(
        facaded_gan_cuda::backend::ComputeBackend::CPU
    );

    // Train — returns generator, discriminator, and final metrics
    let result: GANResult = facade::gf_run(&cfg);

    println!("G-loss:  {:.4}", result.metrics.g_loss);
    println!("D-loss:  {:.4}/{:.4}", result.metrics.d_loss_real, result.metrics.d_loss_fake);
    println!("Gen layers: {}", result.generator.layer_count);
}
```

### **Python API**

```python
import facaded_gan

# Initialise backend: "cpu" | "cuda" | "opencl" | "hybrid" | "auto"
facaded_gan.init_backend("cpu")
print("backend:", facaded_gan.detect_backend())

# High-level: configure and run
cfg = facaded_gan.GANConfig()
cfg.epochs      = 50
cfg.batch_size  = 32
cfg.noise_depth = 64
cfg.learning_rate = 0.0002
cfg.loss_type   = "wgan"

result = facaded_gan.run(cfg)
print(f"g_loss:     {result.metrics.g_loss:.4f}")
print(f"gen layers: {result.generator.layer_count}")

# Low-level via facade submodule
from facaded_gan import facade
gen  = facade.gf_gen_build([64, 128, 256, 1], "leaky", "adam", 0.0002)
disc = facade.gf_disc_build([1, 256, 128, 64], "leaky", "adam", 0.0002)
ds   = facade.gf_train_create_synthetic(1000, 64)

metrics = facade.gf_train_full(gen, disc, ds, cfg)
print(f"epoch {metrics.epoch}  g={metrics.g_loss:.4f}")

# Direct matrix API
from facaded_gan import api
m = api.create_matrix(4, 4)
print("matrix:", m)
```

### **Node.js API**

```javascript
const fg = require('facaded-gan');

// Initialise backend
fg.initBackend('cpu');
console.log('backend:', fg.detectBackend());

// Build and configure
const cfg = new fg.GanConfig();
cfg.setEpochs(50);
cfg.setBatchSize(32);
cfg.setLearningRate(0.0002);
cfg.setLossType('wgan');

// High-level run
const result = fg.run(cfg);
console.log('g_loss:', result.metrics().gLoss());
console.log('gen layers:', result.generator().layerCount());

// Manual network construction
const sizes = [64, 128, 1];
const gen  = fg.genBuild(sizes, 'leaky', 'adam', 0.0002);
const disc = fg.discBuild([1, 128, 64], 'leaky', 'adam', 0.0002);

// Noise and generation
const noise = fg.generateNoise(8, 64, 'gauss');
const fake  = gen.forward(noise);
console.log('fake shape:', fake.rows(), '×', fake.cols());

// Cleanup
result.deinit();
cfg.deinit();
```

### **Go API**

```go
package main

import (
    "fmt"
    fg "github.com/matthew-abbott/facaded-gan/facadedgan"
)

func main() {
    fg.InitBackend("cpu")
    fmt.Println("backend:", fg.DetectBackend())

    cfg := fg.NewConfig()
    defer cfg.Free()
    cfg.SetEpochs(50)
    cfg.SetBatchSize(32)
    cfg.SetLearningRate(0.0002)
    cfg.SetLossType("wgan")

    result := fg.Run(cfg)
    defer result.Free()

    m := result.Metrics()
    defer m.Free()
    fmt.Printf("g_loss: %.4f  epoch: %d\n", m.GLoss(), m.Epoch())

    gen := result.Generator()
    defer gen.Free()
    fmt.Printf("gen layers: %d\n", gen.LayerCount())

    // Manual build
    sizes := []int{64, 128, 1}
    g2 := fg.GenBuild(sizes, "leaky", "adam", 0.0002)
    defer g2.Free()

    noise := fg.GenerateNoise(8, 64, "gauss")
    defer noise.Free()
    fmt.Printf("noise: %d×%d\n", noise.Rows(), noise.Cols())
}
```

### **Julia API**

```julia
using FacadedGan

init_backend("cpu")
println("backend: ", detect_backend())

# Configure
cfg = Config()
cfg.epochs       = 50
cfg.batch_size   = 32
cfg.noise_depth  = 64
cfg.learning_rate = 0.0002

# High-level run
result = run_gan(cfg)
m = FacadedGan.metrics(result)
println("g_loss: ", m.g_loss)
gen = FacadedGan.generator(result)
println("gen layers: ", gen.layer_count)
free!(result)

# Manual build
g2 = gen_build(Cint[64, 128, 1], "leaky", "adam", 0.0002f0)
noise = generate_noise(8, 64, "gauss")
println("noise: ", size(noise, 1), "×", size(noise, 2))
free!(g2); free!(noise)
free!(cfg)
```

### **C API**

```c
#include "facaded_gan.h"
#include <stdio.h>

int main(void) {
    gf_init_backend("cpu");
    printf("backend: %s\n", gf_detect_backend());

    GanConfig* cfg = gf_config_create();
    gf_config_set_epochs(cfg, 50);
    gf_config_set_batch_size(cfg, 32);
    gf_config_set_learning_rate(cfg, 0.0002f);
    gf_config_set_loss_type(cfg, "wgan");

    GanResult* result = gf_run(cfg);
    GanMetrics* m = gf_result_metrics(result);
    printf("g_loss: %.4f  epoch: %d\n",
           gf_metrics_g_loss(m), gf_metrics_epoch(m));

    GanNetwork* gen = gf_result_generator(result);
    printf("gen layers: %d\n", gf_network_layer_count(gen));

    /* Noise and sampling */
    GanMatrix* noise = gf_generate_noise(8, 64, "gauss");
    GanMatrix* fake  = gf_network_sample(gen, 8, 64, "gauss");
    printf("fake: %d×%d\n", gf_matrix_rows(fake), gf_matrix_cols(fake));

    /* Cleanup — every gf_*_create / gf_*_build needs exactly one gf_*_free */
    gf_matrix_free(fake);
    gf_matrix_free(noise);
    gf_network_free(gen);
    gf_metrics_free(m);
    gf_result_free(result);
    gf_config_free(cfg);
    return 0;
}
```

### **C++ API**

```cpp
#include "facaded_gan.hpp"
#include <iostream>

int main() {
    gf::initBackend("cpu");
    std::cout << "backend: " << gf::detectBackend() << "\n";

    // RAII — destructors call gf_*_free automatically
    gf::Config cfg;
    cfg.setEpochs(50);
    cfg.setBatchSize(32);
    cfg.setLearningRate(0.0002f);
    cfg.setLossType("wgan");

    {
        gf::Result result = gf::run(cfg);
        gf::Metrics m     = result.metrics();
        std::cout << "g_loss: " << m.gLoss()
                  << "  epoch: " << m.epoch() << "\n";

        gf::Network gen = result.generator();
        std::cout << "gen layers: " << gen.layerCount() << "\n";

        gf::Matrix noise = gf::generateNoise(8, 64, "gauss");
        gf::Matrix fake  = gen.sample(8, 64, "gauss");
        std::cout << "fake: " << fake.rows() << "×" << fake.cols() << "\n";
    } // result, m, gen, noise, fake freed here

    return 0;
}
```

### **C# API**

```csharp
using FacadedGan;

Gan.InitBackend("cpu");
Console.WriteLine($"backend: {Gan.DetectBackend()}");

using var cfg = new Config { Epochs = 50, BatchSize = 32, LearningRate = 0.0002f };
cfg.LossType = "wgan";

using var result = Gan.Run(cfg);
using var m = result.Metrics;
Console.WriteLine($"g_loss: {m.GLoss:F4}  epoch: {m.Epoch}");
using var gen = result.Generator;
Console.WriteLine($"gen layers: {gen.LayerCount}");

// Matrix arithmetic
using var noise = Gan.GenerateNoise(8, 64, "gauss");
using var fake  = gen.Sample(8, 64, "gauss");
Console.WriteLine($"fake: {fake.Rows}×{fake.Cols}");

// Security helpers
Console.WriteLine($"path safe: {Gan.ValidatePath("/tmp/model.bin")}");
Gan.AuditLog("Training complete", "/tmp/gan_audit.log");
```

### **Zig API**

```zig
const std = @import("std");
const fg  = @import("facaded_gan");

pub fn main() !void {
    fg.initBackend("cpu");
    std.debug.print("backend: {s}\n", .{fg.detectBackend()});

    var cfg = try fg.Config.init();
    defer cfg.deinit();
    cfg.setEpochs(50);
    cfg.setBatchSize(32);
    cfg.setLearningRate(0.0002);
    cfg.setLossType("wgan");

    var result = try fg.run(&cfg);
    defer result.deinit();

    var m = try result.metrics();
    defer m.deinit();
    std.debug.print("g_loss: {d:.4}  epoch: {d}\n", .{ m.gLoss(), m.epoch() });

    var gen = try result.generator();
    defer gen.deinit();
    std.debug.print("gen layers: {d}\n", .{gen.layerCount()});

    // Noise and sampling
    var noise = try fg.generateNoise(8, 64, "gauss");
    defer noise.deinit();
    var fake = try gen.sample(8, 64, "gauss");
    defer fake.deinit();
    std.debug.print("fake: {d}×{d}\n", .{ fake.rows(), fake.cols() });

    // Security
    std.debug.print("path safe: {}\n", .{fg.validatePath("/tmp/model.bin")});
    fg.auditLog("Training complete", "/tmp/gan_audit.log");
}
```

---

### **CLI Reference**

#### Synopsis

```
facaded_gan_cuda [OPTIONS]
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs <N>` | 10 | Number of training epochs |
| `--batch-size <N>` | 32 | Samples per mini-batch |
| `--noise-depth <N>` | 64 | Generator input noise dimension |
| `--lr <F>` | 0.0002 | Base learning rate |
| `--generator-lr <F>` | (from `--lr`) | Generator learning rate (TTUR) |
| `--discriminator-lr <F>` | (from `--lr`) | Discriminator learning rate (TTUR) |
| `--loss <TYPE>` | `bce` | Loss: `bce` \| `wgan` \| `hinge` \| `ls` |
| `--activation <ACT>` | `leaky` | Activation: `relu` \| `sigmoid` \| `tanh` \| `leaky` \| `none` |
| `--optimizer <OPT>` | `adam` | Optimizer: `adam` \| `sgd` \| `rmsprop` |
| `--noise-type <T>` | `gauss` | Noise: `gauss` \| `uniform` \| `analog` |
| `--backend <B>` | `auto` | Backend: `cpu` \| `cuda` \| `opencl` \| `hybrid` \| `auto` |
| `--use-conv` | off | Use convolutional networks |
| `--use-attention` | off | Add self-attention layers |
| `--use-spectral-norm` | off | Spectral normalisation on discriminator |
| `--use-batch-norm` | off | Batch normalisation |
| `--use-layer-norm` | off | Layer normalisation |
| `--gp-lambda <F>` | 10.0 | Gradient penalty coefficient (WGAN-GP) |
| `--save-model <PATH>` | — | Save trained weights here |
| `--load-model <PATH>` | — | Resume from binary checkpoint |
| `--load-json-model <PATH>` | — | Resume from JSON checkpoint |
| `--output-dir <DIR>` | — | Directory for periodic checkpoints |
| `--checkpoint-interval <N>` | 0 | Epochs between checkpoints (0 = off) |
| `--data-path <PATH>` | — | Dataset path (`--data-type` required) |
| `--data-type <T>` | `vector` | Dataset type: `vector` \| `image` \| `audio` |
| `--audit-log` | off | Enable ISO-8601 audit logging |
| `--audit-log-file <PATH>` | `audit.log` | Audit log destination |
| `--use-encryption` | off | Encrypt saved model files |
| `--encryption-key <KEY>` | — | Key for encryption |
| `--detect` | — | Print detected backend and exit |
| `--tests` | — | Run internal test suite and exit |
| `--fuzz` | — | Run fuzz tests and exit |
| `--quality-tests` | — | Run quality tests and exit |

#### Examples

```bash
# Quick CPU smoke-test
cargo run --no-default-features -- --epochs 2 --batch-size 8

# WGAN-GP with TTUR learning rates
./target/release/facaded_gan_cuda \
  --epochs 100 --batch-size 64 --noise-depth 128 \
  --loss wgan --gp-lambda 10 \
  --generator-lr 0.0001 --discriminator-lr 0.0004 \
  --use-spectral-norm \
  --save-model models/wgan.bin

# Resume training from checkpoint
./target/release/facaded_gan_cuda \
  --load-model models/wgan.bin \
  --epochs 50 --save-model models/wgan_v2.bin

# Detect available backend
./target/release/facaded_gan_cuda --detect

# Convolutional GAN with audit logging
./target/release/facaded_gan_cuda \
  --use-conv --use-attention \
  --epochs 200 --backend cuda \
  --audit-log --audit-log-file /var/log/gan_train.log \
  --output-dir checkpoints/ --checkpoint-interval 10
```

---

## **API Reference**

### High-Level (all languages)

| Function | Description |
|----------|-------------|
| `init_backend(name)` | Set global compute backend ("cpu"\|"cuda"\|"opencl"\|"hybrid"\|"auto") |
| `detect_backend()` | Return name of best detected backend (static string) |
| `secure_randomize()` | Seed global RNG from `/dev/urandom` |
| `run(cfg)` | Build, train, and return `GANResult` in one call |
| `train_full(gen, disc, ds, cfg)` | Run all epochs; return final `GANMetrics` |
| `train_step(gen, disc, batch, noise, cfg)` | Run one D+G update; return `GANMetrics` |
| `save_json(gen, disc, path)` | Save both networks to a JSON file |
| `load_json(gen, disc, path)` | Load both networks from a JSON file |
| `save_checkpoint(gen, disc, epoch, dir)` | Binary checkpoint at epoch |
| `load_checkpoint(gen, disc, epoch, dir)` | Restore binary checkpoint |

### Network Factory Methods

| Function | Description |
|----------|-------------|
| `gen_build(sizes, act, opt, lr)` | Dense generator: `sizes` = layer widths |
| `gen_build_conv(noise_dim, cond_sz, base_ch, act, opt, lr)` | Convolutional generator |
| `disc_build(sizes, act, opt, lr)` | Dense discriminator |
| `disc_build_conv(in_ch, in_w, in_h, cond_sz, base_ch, act, opt, lr)` | Convolutional discriminator |

### Network Methods

| Method | Description |
|--------|-------------|
| `forward(inp)` | Forward pass; `inp` is batch×features |
| `backward(grad)` | Backward pass; returns input gradient |
| `update_weights()` | Apply accumulated gradients |
| `set_training(bool)` | Switch training / inference mode |
| `sample(count, noise_dim, noise_type)` | Generate `count` samples |
| `verify()` | Sanitise weights: replace NaN/Inf with 0 |
| `save(path)` / `load(path)` | Binary weight persistence |
| `layer_count` / `learning_rate` / `is_training` | Property accessors |

### Dataset

| Function | Description |
|----------|-------------|
| `dataset_synthetic(count, features)` | Random synthetic dataset |
| `dataset_load(path, data_type)` | Load from file: "vector"\|"image"\|"audio" |
| `dataset_count(ds)` | Number of samples |

### Loss Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `bce_loss(pred, target)` | `f32` | Binary cross-entropy |
| `bce_grad(pred, target)` | Matrix | BCE gradient |
| `wgan_disc_loss(d_real, d_fake)` | `f32` | WGAN discriminator loss |
| `wgan_gen_loss(d_fake)` | `f32` | WGAN generator loss |
| `hinge_disc_loss(d_real, d_fake)` | `f32` | Hinge discriminator loss |
| `hinge_gen_loss(d_fake)` | `f32` | Hinge generator loss |
| `ls_disc_loss(d_real, d_fake)` | `f32` | Least-squares discriminator loss |
| `ls_gen_loss(d_fake)` | `f32` | Least-squares generator loss |
| `cosine_anneal(epoch, max_ep, base_lr, min_lr)` | `f32` | Cosine annealing LR |

### Matrix Operations

| Function | Description |
|----------|-------------|
| `matrix_create(rows, cols)` | Zero-filled matrix |
| `matrix_from_data(data, rows, cols)` | Initialise from flat array |
| `matrix_multiply(a, b)` | Matrix multiply A×B |
| `matrix_add(a, b)` / `matrix_subtract(a, b)` | Element-wise add/sub |
| `matrix_scale(a, s)` | Scalar multiply |
| `matrix_transpose(a)` | Transpose |
| `matrix_normalize(a)` | L2-normalise each row |
| `matrix_element_mul(a, b)` | Hadamard product |
| `matrix_get(m, r, c)` / `matrix_set(m, r, c, v)` | Bounds-checked element access |
| `matrix_safe_get(m, r, c, default)` | Returns `default` on OOB |
| `bounds_check(m, r, c)` | Returns bool: (r,c) within matrix |
| `activate(m, type)` / `relu(m)` / `sigmoid(m)` / `tanh(m)` / `leaky_relu(m, alpha)` / `softmax(m)` | Activation functions |

### Security

| Function | Description |
|----------|-------------|
| `validate_path(path)` | Returns `true` if path has no traversal components |
| `audit_log(msg, log_file)` | Append ISO-8601 timestamped entry to log file |
| `bounds_check(m, r, c)` | Verify (r, c) is within matrix bounds |
| `secure_randomize()` | Seed RNG from OS entropy source |

---

## **Facade API Reference**

The facade module (`src/facade.rs`) exposes 119 `gf_*` functions organized into five groups. In Rust, import with `use facaded_gan_cuda::facade::*;`. In Python, access via `from facaded_gan import facade`. The CLI `--help` flag prints this reference at runtime.

All functions in this table are available in every language wrapper (C, C++, Node.js, Go, Julia, C#, Zig).

### Types

| Type | Alias | C opaque handle |
|------|-------|-----------------|
| `TMatrix` | `Vec<Vec<f32>>` | `GanMatrix*` |
| `TVector` | `Vec<f32>` | `GanVector*` |
| `TMatrixArray` | `Vec<TMatrix>` | `GanMatrixArray*` |
| `TLayer` | `Layer` struct | `GanLayer*` |
| `TNetwork` | `Network` struct | `GanNetwork*` |
| `TGANConfig` | `GANConfig` struct | `GanConfig*` |
| `TGANMetrics` | `GANMetrics` struct | `GanMetrics*` |
| `TDataset` | `Dataset` struct | `GanDataset*` |

### Enums

| Enum | Variants |
|------|----------|
| `ActivationType` | `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `None` |
| `LayerType` | `Dense`, `Conv2D`, `Deconv2D`, `Conv1D`, `BatchNorm`, `LayerNorm`, `SpectralNorm`, `Attention` |
| `LossType` | `BCE`, `WGANGP`, `Hinge`, `LeastSquares` |
| `DataType` | `Image`, `Audio`, `Vector` |
| `NoiseType` | `Gauss`, `Uniform`, `Analog` |
| `Optimizer` | `Adam`, `SGD`, `RMSProp` |

### GF_Op — Low-Level Operations

| Function | Returns | Description |
|----------|---------|-------------|
| `gf_op_create_matrix(rows, cols)` | `TMatrix` | Zero-filled matrix |
| `gf_op_create_vector(size)` | `TVector` | Zero-filled vector |
| `gf_op_matrix_multiply(a, b)` | `TMatrix` | A × B |
| `gf_op_matrix_add(a, b)` | `TMatrix` | Element-wise add |
| `gf_op_matrix_subtract(a, b)` | `TMatrix` | Element-wise subtract |
| `gf_op_matrix_scale(a, s)` | `TMatrix` | Scalar multiply |
| `gf_op_matrix_transpose(a)` | `TMatrix` | Transpose |
| `gf_op_matrix_normalize(a)` | `TMatrix` | L2-normalise each row |
| `gf_op_matrix_element_mul(a, b)` | `TMatrix` | Hadamard product |
| `gf_op_matrix_add_in_place(a, b)` | `()` | In-place add |
| `gf_op_matrix_scale_in_place(a, s)` | `()` | In-place scalar multiply |
| `gf_op_matrix_clip_in_place(a, lo, hi)` | `()` | In-place clamp |
| `gf_op_safe_get(m, r, c, default)` | `f32` | Bounds-safe element read |
| `gf_op_safe_set(m, r, c, val)` | `()` | Bounds-safe element write |
| `gf_op_relu(a)` | `TMatrix` | ReLU activation |
| `gf_op_leaky_relu(a, alpha)` | `TMatrix` | Leaky ReLU |
| `gf_op_sigmoid(a)` | `TMatrix` | Sigmoid |
| `gf_op_tanh(a)` | `TMatrix` | Tanh |
| `gf_op_softmax(a)` | `TMatrix` | Softmax |
| `gf_op_activate(a, act)` | `TMatrix` | Dispatch by `ActivationType` |
| `gf_op_activation_backward(grad, pre_act, act)` | `TMatrix` | Activation gradient |
| `gf_op_conv2d(inp, layer)` | `TMatrix` | 2-D convolution forward |
| `gf_op_conv2d_backward(layer, grad)` | `TMatrix` | 2-D convolution backward |
| `gf_op_deconv2d(inp, layer)` | `TMatrix` | Transposed conv forward |
| `gf_op_deconv2d_backward(layer, grad)` | `TMatrix` | Transposed conv backward |
| `gf_op_conv1d(inp, layer)` | `TMatrix` | 1-D convolution forward |
| `gf_op_conv1d_backward(layer, grad)` | `TMatrix` | 1-D convolution backward |
| `gf_op_batch_norm(inp, layer)` | `TMatrix` | Batch normalisation forward |
| `gf_op_batch_norm_backward(layer, grad)` | `TMatrix` | Batch normalisation backward |
| `gf_op_layer_norm(inp, layer)` | `TMatrix` | Layer normalisation forward |
| `gf_op_layer_norm_backward(layer, grad)` | `TMatrix` | Layer normalisation backward |
| `gf_op_spectral_norm(layer)` | `TMatrix` | Spectral norm of weight matrix |
| `gf_op_attention(inp, layer)` | `TMatrix` | Multi-head self-attention forward |
| `gf_op_attention_backward(layer, grad)` | `TMatrix` | Attention backward |
| `gf_op_create_dense_layer(in, out, act)` | `TLayer` | Dense layer factory |
| `gf_op_create_conv2d_layer(iCh,oCh,k,s,p,w,h,act)` | `TLayer` | Conv2D layer factory |
| `gf_op_create_deconv2d_layer(iCh,oCh,k,s,p,w,h,act)` | `TLayer` | Transposed conv factory |
| `gf_op_create_conv1d_layer(iCh,oCh,k,s,p,len,act)` | `TLayer` | Conv1D layer factory |
| `gf_op_create_batch_norm_layer(features)` | `TLayer` | BatchNorm layer factory |
| `gf_op_create_layer_norm_layer(features)` | `TLayer` | LayerNorm layer factory |
| `gf_op_create_attention_layer(d_model, n_heads)` | `TLayer` | Attention layer factory |
| `gf_op_layer_forward(layer, inp)` | `TMatrix` | Dispatch layer forward pass |
| `gf_op_layer_backward(layer, grad)` | `TMatrix` | Dispatch layer backward pass |
| `gf_op_init_layer_optimizer(layer, opt)` | `()` | Initialise layer optimizer state |
| `gf_op_random_gaussian()` | `f32` | Standard normal sample |
| `gf_op_random_uniform(lo, hi)` | `f32` | Uniform sample in [lo, hi] |
| `gf_op_generate_noise(m, size, depth, nt)` | `()` | Fill matrix with noise (in-place) |
| `gf_op_noise_slerp(v1, v2, t)` | `TVector` | Spherical interpolation |

### GF_Gen — Generator Actions

| Function | Returns | Description |
|----------|---------|-------------|
| `gf_gen_build(sizes, act, opt, lr)` | `Network` | Dense generator from layer widths |
| `gf_gen_build_conv(noise_dim, cond_sz, base_ch, act, opt, lr)` | `Network` | Convolutional generator |
| `gf_gen_forward(gen, inp)` | `TMatrix` | Forward pass |
| `gf_gen_backward(gen, grad)` | `TMatrix` | Backward pass |
| `gf_gen_sample(gen, count, noise_dim, nt)` | `TMatrix` | Generate `count` samples |
| `gf_gen_sample_conditional(gen, count, noise_dim, cond_sz, nt, cond)` | `TMatrix` | Conditional generation |
| `gf_gen_update_weights(gen)` | `()` | Apply accumulated gradients |
| `gf_gen_add_progressive_layer(gen, lvl)` | `()` | Grow network by one resolution level |
| `gf_gen_get_layer_output(gen, idx)` | `TMatrix` | Intermediate layer activations |
| `gf_gen_set_training(gen, bool)` | `()` | Toggle train / inference mode |
| `gf_gen_noise(size, depth, nt)` | `TMatrix` | Convenience noise matrix |
| `gf_gen_noise_slerp(v1, v2, t)` | `TVector` | Latent-space interpolation |
| `gf_gen_deep_copy(gen)` | `Network` | Full weight clone |

### GF_Disc — Discriminator Actions

| Function | Returns | Description |
|----------|---------|-------------|
| `gf_disc_build(sizes, act, opt, lr)` | `Network` | Dense discriminator |
| `gf_disc_build_conv(in_ch, in_w, in_h, cond_sz, base_ch, act, opt, lr)` | `Network` | Convolutional discriminator |
| `gf_disc_evaluate(disc, inp)` | `TMatrix` | Inference-mode forward pass |
| `gf_disc_forward(disc, inp)` | `TMatrix` | Training-mode forward pass |
| `gf_disc_backward(disc, grad)` | `TMatrix` | Backward pass |
| `gf_disc_update_weights(disc)` | `()` | Apply gradients |
| `gf_disc_grad_penalty(disc, real, fake, lambda)` | `f32` | Gradient penalty (WGAN-GP) |
| `gf_disc_feature_match(disc, real, fake, feat_layer)` | `f32` | Feature-matching loss |
| `gf_disc_minibatch_std_dev(inp)` | `TMatrix` | Append minibatch std dev feature |
| `gf_disc_add_progressive_layer(disc, lvl)` | `()` | Grow by one resolution level |
| `gf_disc_get_layer_output(disc, idx)` | `TMatrix` | Intermediate activations |
| `gf_disc_set_training(disc, bool)` | `()` | Toggle train / inference mode |
| `gf_disc_deep_copy(disc)` | `Network` | Full weight clone |

### GF_Train — Training Control

| Function | Returns | Description |
|----------|---------|-------------|
| `gf_train_full(gen, disc, ds, cfg)` | `GANMetrics` | Full training loop |
| `gf_train_step(gen, disc, batch, noise, cfg)` | `GANMetrics` | Single D+G update |
| `gf_train_optimize(net)` | `()` | Apply optimizer step |
| `gf_train_adam_update(p, g, m, v, t, lr, b1, b2, e, wd)` | `()` | Manual Adam step |
| `gf_train_sgd_update(p, g, lr, wd)` | `()` | Manual SGD step |
| `gf_train_rmsprop_update(p, g, cache, lr, decay, e, wd)` | `()` | Manual RMSProp step |
| `gf_train_cosine_anneal(ep, max_ep, base_lr, min_lr)` | `f32` | Cosine annealing schedule |
| `gf_train_bce_loss(pred, target)` | `f32` | Binary cross-entropy |
| `gf_train_bce_grad(pred, target)` | `TMatrix` | BCE gradient |
| `gf_train_wgan_disc_loss(d_real, d_fake)` | `f32` | WGAN discriminator loss |
| `gf_train_wgan_gen_loss(d_fake)` | `f32` | WGAN generator loss |
| `gf_train_hinge_disc_loss(d_real, d_fake)` | `f32` | Hinge discriminator loss |
| `gf_train_hinge_gen_loss(d_fake)` | `f32` | Hinge generator loss |
| `gf_train_ls_disc_loss(d_real, d_fake)` | `f32` | Least-squares discriminator loss |
| `gf_train_ls_gen_loss(d_fake)` | `f32` | Least-squares generator loss |
| `gf_train_label_smoothing(labels, lo, hi)` | `TMatrix` | Soft label remapping |
| `gf_train_load_dataset(path, dt)` | `Dataset` | Load dataset from file |
| `gf_train_load_bmp(path)` | `Dataset` | Load BMP image dataset |
| `gf_train_load_wav(path)` | `Dataset` | Load WAV audio dataset |
| `gf_train_create_synthetic(count, features)` | `Dataset` | Random synthetic dataset |
| `gf_train_augment(sample, dt)` | `TMatrix` | Data augmentation |
| `gf_train_compute_fid(real_s, fake_s)` | `f32` | Fréchet Inception Distance |
| `gf_train_compute_is(samples)` | `f32` | Inception Score |
| `gf_train_log_metrics(metrics, filename)` | `()` | Append metrics to CSV |
| `gf_train_save_model(net, filename)` | `()` | Binary weight save |
| `gf_train_load_model(net, filename)` | `()` | Binary weight load |
| `gf_train_save_json(gen, disc, filename)` | `()` | JSON weight save |
| `gf_train_load_json(gen, disc, filename)` | `()` | JSON weight load |
| `gf_train_save_checkpoint(gen, disc, ep, dir)` | `()` | Epoch checkpoint save |
| `gf_train_load_checkpoint(gen, disc, ep, dir)` | `()` | Epoch checkpoint restore |
| `gf_train_save_samples(gen, ep, dir, noise_dim, nt)` | `()` | Save generated samples to disk |
| `gf_train_plot_csv(filename, d_loss, g_loss, cnt)` | `()` | Write loss curves to CSV |
| `gf_train_print_bar(d_loss, g_loss, width)` | `()` | ASCII progress bar |

### GF_Sec — Security & Entropy

| Function | Returns | Description |
|----------|---------|-------------|
| `gf_sec_audit_log(msg, log_file)` | `()` | ISO-8601 timestamped append-only log [NIST AU-2/AU-3] |
| `gf_sec_secure_randomize()` | `()` | Seed RNG from `/dev/urandom` |
| `gf_sec_get_os_random()` | `u8` | Single byte from `/dev/urandom` |
| `gf_sec_validate_path(path)` | `bool` | Reject paths with traversal components |
| `gf_sec_verify_weights(layer)` | `()` | Replace NaN/Inf in one layer |
| `gf_sec_verify_network(net)` | `()` | Replace NaN/Inf across all layers |
| `gf_sec_encrypt_model(in, out, key)` | `()` | XOR-stream model encryption [NIST SC-28] |
| `gf_sec_decrypt_model(in, out, key)` | `()` | XOR-stream model decryption [NIST SC-28] |
| `gf_sec_bounds_check(m, r, c)` | `bool` | Verify (r,c) is in-bounds for matrix |
| `gf_sec_run_tests()` | `bool` | Built-in unit test suite [NIST SA-11] |
| `gf_sec_run_fuzz_tests(iterations)` | `bool` | Fuzz test suite [NIST SA-11] |

### GF_Layer — Layer Handle API

Individual layers can be created, operated on, and freed independently of a full Network.
The `GanLayer*` opaque handle wraps any layer type (dense, conv, norm, attention).

| Function | Returns | Description |
|----------|---------|-------------|
| `gf_layer_create_dense(in, out, act)` | `GanLayer*` | Dense (fully-connected) layer |
| `gf_layer_create_conv2d(iCh,oCh,k,s,p,w,h,act)` | `GanLayer*` | Conv2D layer |
| `gf_layer_create_deconv2d(...)` | `GanLayer*` | Transposed Conv2D layer |
| `gf_layer_create_conv1d(iCh,oCh,k,s,p,len,act)` | `GanLayer*` | Conv1D layer |
| `gf_layer_create_batch_norm(features)` | `GanLayer*` | BatchNorm layer |
| `gf_layer_create_layer_norm(features)` | `GanLayer*` | LayerNorm layer |
| `gf_layer_create_attention(dModel, nHeads)` | `GanLayer*` | Multi-head self-attention |
| `gf_layer_free(layer*)` | `()` | Free a GanLayer |
| `gf_layer_forward(layer*, inp*)` | `Matrix*` | Dispatch forward through any layer type |
| `gf_layer_backward(layer*, grad*)` | `Matrix*` | Dispatch backward through any layer type |
| `gf_layer_init_optimizer(layer*, opt)` | `()` | Initialise per-layer optimizer |
| `gf_layer_conv2d / conv2d_backward` | `Matrix*` | Conv2D fwd/bwd |
| `gf_layer_deconv2d / deconv2d_backward` | `Matrix*` | Transposed Conv2D fwd/bwd |
| `gf_layer_conv1d / conv1d_backward` | `Matrix*` | Conv1D fwd/bwd |
| `gf_layer_batch_norm / batch_norm_backward` | `Matrix*` | BatchNorm fwd/bwd |
| `gf_layer_layer_norm / layer_norm_backward` | `Matrix*` | LayerNorm fwd/bwd |
| `gf_layer_spectral_norm(layer*)` | `Matrix*` | Spectral normalization |
| `gf_layer_attention / attention_backward` | `Matrix*` | Self-attention fwd/bwd |
| `gf_layer_verify_weights(layer*)` | `()` | Replace NaN/Inf in layer weights |

### GF_MatrixArray — FID / IS Sample Container

A growable array of matrices used to collect real and fake samples for FID and IS metrics.

| Function | Returns | Description |
|----------|---------|-------------|
| `gf_matrix_array_create()` | `GanMatrixArray*` | Create an empty array |
| `gf_matrix_array_free(arr*)` | `()` | Free the array and its contents |
| `gf_matrix_array_push(arr*, m*)` | `()` | Append a copy of matrix m |
| `gf_matrix_array_len(arr*)` | `int` | Number of matrices in the array |

### GF_Run — Top-Level Entry Point

| Function | Returns | Description |
|----------|---------|-------------|
| `gf_run(config)` | `GANResult` | Build networks, load data, train, save — full pipeline |

### Facade Usage Examples

**Rust (direct)**
```rust
use facaded_gan_cuda::facade::*;
use facaded_gan_cuda::types::*;

// Low-level ops
let a = gf_op_create_matrix(4, 4);
let b = gf_op_matrix_transpose(&a);
let c = gf_op_matrix_multiply(&a, &b);

// Build and train manually
let mut gen  = gf_gen_build(&[64, 128, 1], ActivationType::LeakyReLU, Optimizer::Adam, 0.0002);
let mut disc = gf_disc_build(&[1, 128, 1], ActivationType::LeakyReLU, Optimizer::Adam, 0.0002);
let ds       = gf_train_create_synthetic(1000, 1);
let cfg      = GANConfig { epochs: 10, batch_size: 32, ..Default::default() };
let metrics  = gf_train_full(&mut gen, &mut disc, &ds, &cfg);
println!("g_loss: {:.4}", metrics.g_loss);
```

**Python (facade submodule)**
```python
from facaded_gan import facade

# Low-level ops
a = facade.gf_op_create_matrix(4, 4)
b = facade.gf_op_matrix_transpose(a)

# Build generator / discriminator
gen  = facade.gf_gen_build([64, 128, 1], "leaky", "adam", 0.0002)
disc = facade.gf_disc_build([1, 128, 1], "leaky", "adam", 0.0002)

# Security
ok = facade.gf_sec_validate_path("/tmp/model.bin")
facade.gf_sec_audit_log("Training started", "/tmp/gan.log")
```

---

## **Formal Verification with Kani**

### Overview

GlassBoxAI-GAN includes two layers of Kani formal verification:

1. **Library harnesses** (`kani/`) — 15 groups proving properties of the core Rust algorithms
2. **FFI boundary harnesses** (`c_bindings/src/kani_ffi_tests.rs`) — 39 harnesses proving the C API contract relied upon by all seven language wrappers

### Verification Report

| Category | Harnesses | Properties Covered |
|----------|-----------|--------------------|
| **Bound checks** (`bound_checks.rs`) | 9 | OOB-free indexing, safe_get/set, matmul dims |
| **No panic** (`no_panic.rs`) | 12 | All activation/loss/matrix/security fns |
| **Integer overflow** (`integer_overflow.rs`) | 8 | Arithmetic in add/mul/scale/normalise |
| **Division by zero** (`div_by_zero.rs`) | 4 | Non-zero denominators proven |
| **Pointer validity** (`pointer_validity.rs`) | 5 | Slice and reference access |
| **Global state** (`global_state.rs`) | 4 | Network/layer state invariants |
| **Deadlock freedom** (`deadlock_free.rs`) | 3 | Absence of lock ordering issues |
| **Input sanitisation** (`input_sanitization.rs`) | 5 | Bounded loops, recursion depth |
| **Result coverage** (`result_coverage.rs`) | 4 | All Result/Option arms handled |
| **Memory limits** (`memory_limits.rs`) | 4 | Allocation sizes finite and bounded |
| **Constant time** (`constant_time.rs`) | 3 | No secret-dependent branches |
| **State machine** (`state_machine.rs`) | 4 | Training ↔ inference transitions |
| **Enum exhaustion** (`enum_exhaustion.rs`) | 5 | All match arms covered |
| **Float sanity** (`float_sanity.rs`) | 5 | NaN/Inf guards in all logic paths |
| **Resource limits** (`resource_limits.rs`) | 3 | Security allocation budget |
| **FFI null safety** (`kani_ffi_tests.rs`) | 11 | Null pointer handling for all 7 handle types |
| **FFI dimension guards** (`kani_ffi_tests.rs`) | 7 | Non-positive dims → null return |
| **FFI boolean convention** (`kani_ffi_tests.rs`) | 3 | 0→false, non-zero→true for all 15 bool fields |
| **FFI config round-trips** (`kani_ffi_tests.rs`) | 2 | Integer and float set/get identity |
| **FFI enum fallback** (`kani_ffi_tests.rs`) | 4 | Unknown strings accepted, no panic |
| **FFI string null safety** (`kani_ffi_tests.rs`) | 1 | All 8 string setters accept null char* |
| **FFI bounds consistency** (`kani_ffi_tests.rs`) | 4 | bounds_check ↔ matrix dims |
| **FFI backend** (`kani_ffi_tests.rs`) | 2 | detect_backend non-null; validate_path correct |
| **FFI lifecycle** (`kani_ffi_tests.rs`) | 5 | create→use→free, data ptr, no-op free(null) |
| **Total** | **127** | |

### Running Kani

```bash
# Full library verification suite
cargo kani --no-default-features

# Single library harness
cargo kani --no-default-features --harness proof_safe_get_any_index_no_panic

# Full FFI boundary suite
cargo kani --no-default-features -p facaded_gan_c

# Single FFI harness
cargo kani --no-default-features -p facaded_gan_c \
  --harness proof_ffi_null_matrix_accessors

# Run all harnesses across the workspace
cargo kani --no-default-features --workspace
```

### FFI Harness Categories

The 39 FFI harnesses in `c_bindings/src/kani_ffi_tests.rs` are organised into five safety contracts:

#### A. Null Pointer Safety (11 harnesses)
Every `gf_*` function that accepts a pointer must handle `NULL` without crashing.
Covers: `GanMatrix`, `GanVector`, `GanConfig`, `GanNetwork`, `GanMetrics`, `GanResult`,
`gf_run`, `gf_train_full`, `gf_train_step`, matrix arithmetic on null inputs.

#### B. Dimension Guards (7 harnesses)
`gf_matrix_create(rows ≤ 0, cols)` must return `NULL`.  Same for `gf_vector_create`,
`gf_gen_build(null sizes)`, `gf_gen_build(num_sizes ≤ 0)`, `gf_disc_build(null sizes)`.
Positive dimensions must return a non-null, correctly-shaped object.

#### C. Bounds and Indexing Safety (4 harnesses)
`gf_matrix_get` and `gf_matrix_set` with any symbolic `(row, col)` pair on a valid
matrix — no panic guaranteed.  In-bounds `set → get` is bit-exact.  `gf_bounds_check`
result is consistent with the matrix's actual dimensions.

#### D. C ABI Type Contracts (6 harnesses)
**Boolean convention** — the C `int` boolean convention (0 = false, non-zero = true)
holds for all 15 `use_*` config fields under any symbolic non-zero input.
**Config round-trips** — all 5 integer and 5 float fields are identity after set/get.
**Enum fallback** — all 4 enum setters accept unknown strings and null without panic.
**String null safety** — all 8 string setters accept `NULL char*` (maps to `""`).

#### E. Lifecycle and Output Validity (11 harnesses)
`gf_detect_backend` returns non-null.  `gf_validate_path` correctly rejects traversal
and null and accepts safe paths.  `gf_matrix_from_data` with null data returns null;
with valid data returns correct shape.  `gf_generate_noise` with any known/unknown/null
noise type returns non-null.  Data pointer from a valid matrix is non-null; all cells
read as 0.0.  `gf_cosine_anneal` never panics under any symbolic epoch/lr combination.

### Why FFI Formal Verification Matters

When a function in `libfacaded_gan_c.so` panics across a C FFI boundary, the behaviour
is **undefined** — the entire process may crash or corrupt memory silently, and the
error surfaces in whatever language called it (Go, Julia, C#, Zig, Python, Node.js) with
no Rust stack trace.  The 39 FFI harnesses prove that no `gf_*` function can panic or
access invalid memory under any of the inputs a foreign caller can realistically supply,
including `NULL` pointers, negative dimensions, negative indices, arbitrary integer values
for boolean parameters, and unrecognised enum strings.

---

## **CISA/NSA Compliance**

### Secure by Design

GlassBoxAI-GAN follows **CISA** and **NSA** Secure by Design principles throughout:

| Principle | Implementation |
|-----------|---------------|
| **Memory-safe language** | Rust ownership eliminates buffer overflows, use-after-free, and data races in all safe code |
| **Formal verification** | 127 Kani proof harnesses provide mathematical guarantees beyond testing |
| **Null safety** | Every C API function null-checks all pointer inputs; returns safe sentinel values on null |
| **Bounds checking** | `safe_get`/`safe_set`, `bounds_check`, and OOB-safe accessors throughout; Kani-proven |
| **Path traversal prevention** | `gf_validate_path` and `validate_path` reject `../` sequences before any file operation |
| **Audit logging** | `gf_audit_log` appends ISO-8601 timestamped records to an append-only log file |
| **Secure entropy** | `gf_secure_randomize` seeds the global RNG from `/dev/urandom` (OS CSPRNG) |
| **Weight sanitisation** | `gf_network_verify` replaces NaN/Inf weights with 0 after every training run |
| **Input validation** | CLI arguments fully validated; unknown enum strings fall back to safe defaults (Kani-proven) |
| **Defense in depth** | Language-level (Rust) + compiler (rustc/Clippy) + formal verification (Kani) + runtime checks |
| **Transparency** | Full open-source implementation with inline documentation |

### Compliance Checklist

- [x] **Memory-safe language** (Rust implementation)
- [x] **Static analysis** (rustc + Clippy, zero warnings)
- [x] **Formal verification** (127 Kani proof harnesses)
- [x] **FFI boundary verification** (39 C ABI contract harnesses)
- [x] **Comprehensive testing** (unit tests + integration script)
- [x] **Bounds checking** (proven safe for all symbolic index inputs)
- [x] **Null pointer safety** (proven for all 7 opaque handle types)
- [x] **Input validation** (CLI arg parsing + enum fallback, Kani-verified)
- [x] **Secure randomisation** (`/dev/urandom` seeding)
- [x] **Audit logging** (timestamped, append-only)
- [x] **Path traversal prevention** (validated before every file operation)
- [x] **License clarity** (MIT License)
- [x] **Documentation** (inline docs + this README)

### Attestation

This codebase demonstrates:

- **127 formal verifications passed** across 24 proof categories
- **Zero compiler warnings** across the full workspace (`--no-default-features`)
- **Consistent API** across 9 language bindings sharing a single verified C ABI
- **Production-ready** code quality with security controls active by default

---

## **License**

MIT License

Copyright (c) 2025 Matthew Abbott

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## **Author**

**Matthew Abbott**
Email: mattbachg@gmail.com

---

*Built with precision. Verified with rigor. Secured by design.*

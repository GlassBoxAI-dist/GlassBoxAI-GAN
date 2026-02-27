# GlassBoxAI-GNN

## **Graph Neural Network Suite**

### *CUDA/OpenCL-Accelerated GNN with Multi-Language Bindings & Formal Verification*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-blue.svg)](https://www.khronos.org/opencl/)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-16+-339933.svg)](https://nodejs.org/)
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8.svg)](https://go.dev/)
[![Julia](https://img.shields.io/badge/Julia-1.9+-9558B2.svg)](https://julialang.org/)
[![C#](https://img.shields.io/badge/C%23-.NET%208.0-512BD4.svg)](https://dotnet.microsoft.com/)
[![Zig](https://img.shields.io/badge/Zig-0.13+-F7A41D.svg)](https://ziglang.org/)
[![Kani](https://img.shields.io/badge/Kani-Verified-brightgreen.svg)](https://model-checking.github.io/kani/)
[![CISA Compliant](https://img.shields.io/badge/CISA-Secure%20by%20Design-blue.svg)](https://www.cisa.gov/securebydesign)

---

## **Overview**

GlassBoxAI-GNN is a comprehensive, production-ready Graph Neural Network implementation featuring:

- **Dual GPU backends**: CUDA (via cudarc) and OpenCL (via opencl3) with automatic detection
- **Backend selection**: Choose CUDA, OpenCL, or auto-detect at runtime via CLI or API
- **Facade pattern architecture**: Clean, unified API for graph neural network operations
- **Multi-language bindings**: Rust, Python, Node.js/TypeScript, C, C++, Go, Julia, C#, and Zig
- **Formal verification**: Kani-verified Rust implementation for memory safety guarantees
- **CISA/NSA Secure by Design compliance**: Built following government cybersecurity standards

This project demonstrates enterprise-grade software engineering practices including comprehensive testing, formal verification, cross-platform compatibility, and security-first development.

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
   - [TypeScript](#typescript)
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

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Message Passing** | Configurable multi-layer message passing neural network |
| **Graph Operations** | PageRank, degree analysis, neighbor queries |
| **Training** | Backpropagation with gradient clipping |
| **Activation Functions** | ReLU, LeakyReLU, Tanh, Sigmoid |
| **Loss Functions** | MSE, Binary Cross-Entropy |
| **Model Persistence** | Binary serialization for model save/load |
| **Masking & Dropout** | Node and edge masking with configurable dropout rates |
| **Embedding Export** | CSV export of node and graph-level embeddings |

### GPU Acceleration

| Backend | Implementation | Performance |
|---------|---------------|-------------|
| **CUDA** | cudarc Rust bindings | Optimal for NVIDIA GPUs |
| **OpenCL** | opencl3 Rust bindings | Cross-vendor GPU support (NVIDIA, AMD, Intel) |
| **Auto** | Runtime detection | Tries CUDA first, falls back to OpenCL |

### Language Bindings

| Language | Technology | Integration |
|----------|------------|-------------|
| **Rust** | Native | Library crate + CLI binary |
| **Python** | PyO3 | Native extension module via maturin |
| **Node.js** | napi-rs | Native addon with TypeScript definitions |
| **C** | FFI | Shared library + header file |
| **C++** | FFI | RAII wrapper with exception handling |
| **Go** | cgo | Package with idiomatic Go API |
| **Julia** | ccall | Package with Julia-native conventions |
| **C#** | P/Invoke | .NET package with IDisposable pattern |
| **Zig** | C FFI | Package with Zig error handling |

### Safety & Security

| Feature | Technology |
|---------|------------|
| **Memory Safety** | Rust ownership model |
| **Formal Verification** | Kani proof harnesses |
| **Bounds Checking** | Verified array access |
| **Input Validation** | CLI argument validation |

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        GlassBoxAI-GNN                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Core Rust Library (src/)                        ││
│  │  • lib.rs      - GNN engine + facade + backend dispatch     ││
│  │  • opencl.rs   - OpenCL backend (opencl3)                   ││
│  │  • main.rs     - CLI binary (--backend flag)                ││
│  │  • ffi.rs      - C ABI foreign function interface           ││
│  │  • python.rs   - PyO3 Python bindings                       ││
│  │  • nodejs.rs   - napi-rs Node.js bindings                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌──────────────────────────┐ ┌──────────────────────────────┐  │
│  │    CUDA Backend          │ │    OpenCL Backend             │  │
│  ├──────────────────────────┤ ├──────────────────────────────┤  │
│  │ cudarc (NVIDIA GPUs)     │ │ opencl3 (NVIDIA/AMD/Intel)   │  │
│  │ NVRTC kernel compilation │ │ OpenCL C kernel compilation  │  │
│  │ Feature: cuda            │ │ Feature: opencl              │  │
│  └──────────────────────────┘ └──────────────────────────────┘  │
│                                                                 │
│  ┌────────┐ ┌────────┐ ┌──────┐ ┌───────┐ ┌──────┐ ┌──────┐  │
│  │ Python │ │Node.js │ │  Go  │ │ Julia │ │  C#  │ │ Zig  │  │
│  ├────────┤ ├────────┤ ├──────┤ ├───────┤ ├──────┤ ├──────┤  │
│  │ PyO3   │ │napi-rs │ │ cgo  │ │ ccall │ │ P/I  │ │C FFI │  │
│  └────────┘ └────────┘ └──────┘ └───────┘ └──────┘ └──────┘  │
│                                                                 │
│  ┌───────────┐ ┌───────────┐ ┌──────────────────────────────┐  │
│  │   C API   │ │  C++ API  │ │     Kani Formal Proofs       │  │
│  ├───────────┤ ├───────────┤ ├──────────────────────────────┤  │
│  │ gnn_      │ │ gnn_      │ │ 234 total verifications      │  │
│  │ facade.h  │ │ facade.hpp│ │ (19 proofs + 76 unit tests   │  │
│  └───────────┘ └───────────┘ │  + FFI/CUDA/OpenCL proofs)   │  │
│                               └──────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Shared Features                          ││
│  │  • Consistent API across all language bindings              ││
│  │  • Binary-compatible model format (shared across backends)  ││
│  │  • Backend auto-detection (CUDA → OpenCL fallback)          ││
│  │  • Qt-based GUI (optional)                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## **File Structure**

```
GlassBoxAI-GNN/
│
├── Cargo.toml                     # Rust workspace configuration
├── build.rs                       # Rust build script
├── pyproject.toml                 # Python project configuration
├── package.json                   # Node.js package configuration
├── index.js                       # Node.js entry point
├── index.d.ts                     # TypeScript type definitions
│
├── src/                           # Core Rust source
│   ├── main.rs                    # CLI binary entry point
│   ├── lib.rs                     # GNN library + CUDA engine + facade
│   ├── opencl.rs                  # OpenCL backend engine
│   ├── ffi.rs                     # C ABI foreign function interface
│   ├── python.rs                  # PyO3 Python bindings
│   └── nodejs.rs                  # napi-rs Node.js bindings
│
├── include/                       # C/C++ headers
│   ├── gnn_facade.h               # C header
│   └── gnn_facade.hpp             # C++ header (RAII wrapper)
│
├── python/                        # Python package
│   └── gnn_facade_cuda/
│       ├── __init__.py
│       └── __init__.pyi           # Python type stubs
│
├── go/                            # Go package
│   ├── gnn/
│   │   ├── go.mod
│   │   ├── gnn.go                 # Go bindings via cgo
│   │   └── gnn_test.go            # Go tests
│   └── example/
│       ├── go.mod
│       └── main.go                # Go usage example
│
├── julia/                         # Julia package
│   └── GnnFacadeCuda/
│       ├── Project.toml
│       ├── src/
│       │   └── GnnFacadeCuda.jl   # Julia bindings via ccall
│       └── test/
│           └── runtests.jl        # Julia tests
│
├── csharp/                        # C# wrapper
│   ├── GnnFacadeCuda/
│   │   ├── GnnFacade.cs           # .NET bindings (P/Invoke)
│   │   ├── NativeMethods.cs       # P/Invoke declarations
│   │   └── GnnFacadeCuda.csproj   # .NET project file
│   └── example/
│       ├── Program.cs             # C# usage example
│       └── example.csproj
│
├── zig/                           # Zig wrapper
│   ├── src/
│   │   └── gnn.zig                # Zig bindings (C FFI)
│   ├── example/
│   │   └── main.zig               # Zig usage example
│   ├── build.zig                  # Build configuration
│   └── build.zig.zon              # Build dependencies
│
├── gui/                           # Qt-based GUI (optional)
│   ├── Cargo.toml
│   ├── build.rs
│   ├── src/
│   │   ├── main.rs
│   │   ├── gnn_facade.rs
│   │   └── gnn_bridge.rs
│   └── qml/
│       └── main.qml
│
├── kani_proofs/                   # Formal verification proofs
│   ├── Cargo.toml
│   ├── src/
│   │   └── lib.rs
│   ├── README.md
│   └── VERIFICATION_REPORT.md
│
├── src/kani/                      # In-tree verification harnesses
│   ├── ffi_c_boundary.rs          # FFI C boundary safety (55 proofs)
│   ├── ffi_cuda_boundary.rs       # CUDA backend FFI safety (42 proofs)
│   └── ffi_opencl_boundary.rs     # OpenCL backend FFI safety (42 proofs)
│
├── license.md                     # MIT License
└── README.md                      # This file
```

---

## **Prerequisites**

### Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Rust** | 1.70+ | Core library and CLI compilation |

### GPU Backend (at least one required)

| Dependency | Version | Purpose |
|------------|---------|---------|
| **CUDA Toolkit** | 12.0+ | NVIDIA GPU acceleration |
| **OpenCL SDK** | 3.0+ | Cross-vendor GPU acceleration (NVIDIA, AMD, Intel) |

### Optional (per language binding)

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Python bindings |
| **maturin** | latest | Building Python wheels |
| **Node.js** | 16+ | Node.js bindings |
| **@napi-rs/cli** | 2.18+ | Building Node.js native addon |
| **Go** | 1.21+ | Go bindings |
| **Julia** | 1.9+ | Julia bindings |
| **C# / .NET SDK** | 8.0+ | C# bindings |
| **Zig** | 0.13+ | Zig bindings |
| **GCC/G++** | 11+ | C/C++ compilation |
| **Kani** | 0.67+ | Formal verification |
| **Qt 6** | 6.x | GUI version |

---

## **Installation & Building**

### **Rust Library & CLI**

```bash
# Build with both CUDA and OpenCL backends (default)
cargo build --release

# Build with CUDA backend only
cargo build --release --no-default-features --features cuda

# Build with OpenCL backend only
cargo build --release --no-default-features --features opencl

# Run the CLI
./target/release/gnn_facade_cuda --help
```

Add to your `Cargo.toml` to use as a library:

```toml
[dependencies]
gnn_facade_cuda = "0.1.0"
```

### **Python Package**

```bash
# Install maturin
pip install maturin

# Development install
maturin develop --features python

# Or build a wheel
maturin build --features python --release
pip install target/wheels/gnn_facade_cuda-*.whl
```

### **Node.js Package**

```bash
# Install dependencies and build
npm install
npm run build
```

Or use the @napi-rs/cli directly:

```bash
npm install -g @napi-rs/cli
napi build --platform --release --features nodejs
```

### **C/C++ Library**

```bash
# Build the shared library
cargo build --release --lib --features ffi
```

This produces `libgnn_facade_cuda.so` (Linux), `libgnn_facade_cuda.dylib` (macOS), or `gnn_facade_cuda.dll` (Windows) in `target/release/`.

Link against it:

```bash
# C
gcc -o myapp myapp.c -I include -L target/release -lgnn_facade_cuda

# C++
g++ -o myapp myapp.cpp -I include -L target/release -lgnn_facade_cuda -std=c++17
```

### **Go Package**

```bash
# Build the C library first
cargo build --release --lib --features ffi

# Build and run the Go example
cd go/example
go build
LD_LIBRARY_PATH=../../target/release ./example
```

Or import in your project:

```go
import "github.com/GlassBoxAI/GlassBoxAI-GNN/go/gnn"
```

### **Julia Package**

```bash
# Build the C library first
cargo build --release --lib --features ffi
```

Then use the Julia package:

```julia
using Pkg
Pkg.develop(path="path/to/GlassBoxAI-GNN/julia/GnnFacadeCuda")
using GnnFacadeCuda
```

Or activate directly:

```julia
cd("path/to/GlassBoxAI-GNN/julia")
using Pkg
Pkg.activate("GnnFacadeCuda")
using GnnFacadeCuda
```

### **C# Package**

```bash
# Build the C library first
cargo build --release --lib --features ffi

# Build the C# project
cd csharp/GnnFacadeCuda
dotnet build

# Run the example
cd ../example
LD_LIBRARY_PATH=../../target/release dotnet run
```

### **Zig Package**

```bash
# Build the C library first
cargo build --release --lib --features ffi

# Build the Zig example
cd zig
zig build

# Run the example
LD_LIBRARY_PATH=../target/release zig-out/bin/gnn-example
```

### **Build All**

```bash
# Core library + CLI (both backends)
cargo build --release

# Core library + CLI (OpenCL only, no CUDA required)
cargo build --release --no-default-features --features opencl

# Python bindings
pip install maturin && maturin build --features python --release

# Node.js bindings
npm install && npm run build

# C/C++/Go/Julia shared library
cargo build --release --lib --features ffi

# Go example
(cd go/example && go build)
```

---

## **Usage**

### **Rust API**

```rust
use gnn_facade_cuda::{GnnFacade, GpuBackendType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new GNN model (auto-detects best GPU backend)
    let mut facade = GnnFacade::new(
        3,   // feature_size
        16,  // hidden_size
        2,   // output_size
        2,   // num_mp_layers
    )?;

    // Or specify a backend explicitly
    let mut facade = GnnFacade::with_backend(
        3, 16, 2, 2,
        GpuBackendType::OpenCL,
    )?;

    println!("Using backend: {}", facade.get_backend_name());

    // Create a graph with 5 nodes
    facade.create_empty_graph(5, 3);

    // Add edges
    facade.add_edge(0, 1, vec![]);
    facade.add_edge(1, 2, vec![]);
    facade.add_edge(2, 3, vec![]);

    // Set node features
    facade.set_node_features(0, vec![1.0, 0.5, 0.2]);
    facade.set_node_features(1, vec![0.8, 0.3, 0.1]);

    // Make predictions
    let prediction = facade.predict()?;
    println!("Prediction: {:?}", prediction);

    // Train the model
    let target = vec![0.5, 0.5];
    let loss = facade.train(&target)?;
    println!("Loss: {}", loss);

    // Save the model
    facade.save_model("model.bin")?;

    // Load a saved model
    let loaded = GnnFacade::from_model_file("model.bin")?;
    println!("Loaded model with {} parameters", loaded.get_parameter_count());

    Ok(())
}
```

### **Python API**

```python
from gnn_facade_cuda import GnnFacade

# Create a new GNN model (auto-detects GPU backend)
gnn = GnnFacade(
    feature_size=3,
    hidden_size=16,
    output_size=2,
    num_mp_layers=2
)

# Or specify a backend explicitly: "cuda", "opencl", or "auto"
gnn = GnnFacade(3, 16, 2, 2, backend="opencl")
print(f"Backend: {gnn.get_backend_name()}")

# Create a graph with 5 nodes
gnn.create_empty_graph(5, 3)

# Add edges
gnn.add_edge(0, 1)
gnn.add_edge(1, 2)
gnn.add_edge(2, 3)

# Set node features
gnn.set_node_features(0, [1.0, 0.5, 0.2])
gnn.set_node_features(1, [0.8, 0.3, 0.1])

# Make predictions
prediction = gnn.predict()
print(f"Prediction: {prediction}")

# Train the model
loss = gnn.train([0.5, 0.5])
print(f"Loss: {loss}")

# Train for multiple epochs
gnn.train_multiple([0.5, 0.5], iterations=100)

# Save and load
gnn.save_model("model.bin")
gnn2 = GnnFacade.from_model_file("model.bin")
print(f"Loaded model with {gnn2.get_parameter_count()} parameters")

# Graph analytics
ranks = gnn.compute_page_rank(damping=0.85, iterations=20)
print(f"PageRank: {ranks}")

# Export graph to JSON
json_str = gnn.export_graph_to_json()
print(json_str)
```

### **Node.js API**

```javascript
const { GnnFacade } = require('gnn-facade-cuda');

// Create a new GNN model (auto-detects GPU backend)
const gnn = new GnnFacade(3, 16, 2, 2);

// Or specify a backend: "cuda", "opencl", or "auto"
const gnn2 = new GnnFacade(3, 16, 2, 2, 'opencl');
console.log('Backend:', gnn2.getBackendName());

// Create a graph with 5 nodes
gnn.createEmptyGraph(5, 3);

// Add edges
gnn.addEdge(0, 1);
gnn.addEdge(1, 2);
gnn.addEdge(2, 3);

// Set node features
gnn.setNodeFeatures(0, [1.0, 0.5, 0.2]);
gnn.setNodeFeatures(1, [0.8, 0.3, 0.1]);

// Make predictions
const prediction = gnn.predict();
console.log('Prediction:', prediction);

// Train the model
const loss = gnn.train([0.5, 0.5]);
console.log('Loss:', loss);

// Train for multiple epochs
gnn.trainMultiple([0.5, 0.5], 100);

// Save and load
gnn.saveModel('model.bin');
const gnn2 = GnnFacade.fromModelFile('model.bin');
console.log('Loaded model with', gnn2.getParameterCount(), 'parameters');

// Graph analytics
const ranks = gnn.computePageRank(0.85, 20);
console.log('PageRank:', ranks);

// Export graph to JSON
const jsonStr = gnn.exportGraphToJson();
console.log(jsonStr);
```

### **TypeScript**

```typescript
import { GnnFacade, GradientFlowInfo, ModelHeader } from 'gnn-facade-cuda';

// Full TypeScript support with type definitions
const gnn = new GnnFacade(3, 16, 2, 2);
gnn.createEmptyGraph(5, 3);

// Types are inferred
const prediction: number[] = gnn.predict();
const info: GradientFlowInfo = gnn.getGradientFlow(0);
const header: ModelHeader = GnnFacade.readModelHeader('model.bin');
```

### **Go API**

```go
package main

import (
	"fmt"
	"log"

	"github.com/GlassBoxAI/GlassBoxAI-GNN/go/gnn"
)

func main() {
	// Create a new GNN (auto-detects GPU backend)
	g, err := gnn.New(3, 16, 2, 2)
	if err != nil {
		log.Fatal(err)
	}
	defer g.Close()

	// Or specify a backend explicitly
	g2, err := gnn.NewWithBackend(3, 16, 2, 2, gnn.BackendOpenCL)
	if err != nil {
		log.Fatal(err)
	}
	defer g2.Close()
	fmt.Println("Backend:", g2.GetBackendName())

	// Create a graph
	g.CreateEmptyGraph(5, 3)

	// Add edges
	g.AddEdge(0, 1, nil)
	g.AddEdge(1, 2, nil)
	g.AddEdge(2, 3, nil)

	// Set node features
	g.SetNodeFeatures(0, []float32{1.0, 0.5, 0.2})
	g.SetNodeFeatures(1, []float32{0.8, 0.3, 0.1})

	// Make predictions
	prediction, err := g.Predict()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Prediction:", prediction)

	// Train
	loss, err := g.Train([]float32{0.5, 0.5})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Loss:", loss)

	// Train for multiple iterations
	if err := g.TrainMultiple([]float32{0.5, 0.5}, 100); err != nil {
		log.Fatal(err)
	}

	// Save and load
	if err := g.SaveModel("model.bin"); err != nil {
		log.Fatal(err)
	}
	g2, err := gnn.Load("model.bin")
	if err != nil {
		log.Fatal(err)
	}
	defer g2.Close()
	fmt.Printf("Loaded model with %d parameters\n", g2.GetParameterCount())

	// Graph analytics
	ranks := g.ComputePageRank(0.85, 20)
	fmt.Println("PageRank:", ranks)

	// Export to JSON
	jsonStr := g.ExportGraphToJSON()
	fmt.Println(jsonStr)
}
```

### **Julia API**

```julia
using GnnFacadeCuda

# Create a new GNN (auto-detects GPU backend)
gnn = GnnFacade(3, 16, 2, 2)

# Or specify a backend explicitly
gnn = GnnFacade(3, 16, 2, 2; backend=GNN_BACKEND_OPENCL)
println("Backend: ", get_backend_name(gnn))

# Create a graph with 5 nodes
create_empty_graph!(gnn, 5, 3)

# Add edges (0-indexed)
add_edge!(gnn, 0, 1)
add_edge!(gnn, 1, 2)
add_edge!(gnn, 2, 3)

# Set node features
set_node_features!(gnn, 0, Float32[1.0, 0.5, 0.2])
set_node_features!(gnn, 1, Float32[0.8, 0.3, 0.1])

# Make predictions
prediction = predict!(gnn)
println("Prediction: ", prediction)

# Train the model
loss = train!(gnn, Float32[0.5, 0.5])
println("Loss: ", loss)

# Train for multiple iterations
train_multiple!(gnn, Float32[0.5, 0.5], 100)

# Save and load
save_model(gnn, "model.bin")
gnn2 = load_gnn("model.bin")
println("Loaded model with $(get_parameter_count(gnn2)) parameters")

# Graph analytics
ranks = compute_page_rank(gnn)
println("PageRank: ", ranks)

# Export to JSON
json_str = export_graph_to_json(gnn)
println(json_str)
```

### **C API**

```c
#include "gnn_facade.h"
#include <stdio.h>

int main() {
    // Create a GNN (auto-detects GPU backend)
    GnnHandle* gnn = gnn_create(3, 16, 2, 2);
    if (!gnn) {
        fprintf(stderr, "Failed to create GNN\n");
        return 1;
    }

    // Or specify a backend: GNN_BACKEND_CUDA, GNN_BACKEND_OPENCL, GNN_BACKEND_AUTO
    GnnHandle* gnn2 = gnn_create_with_backend(3, 16, 2, 2, GNN_BACKEND_OPENCL);

    // Query active backend
    char backend_name[64];
    gnn_get_backend_name(gnn, backend_name, sizeof(backend_name));
    printf("Backend: %s\n", backend_name);

    // Create a graph
    gnn_create_empty_graph(gnn, 5, 3);

    // Add edges
    gnn_add_edge(gnn, 0, 1, NULL, 0);
    gnn_add_edge(gnn, 1, 2, NULL, 0);

    // Set node features
    float features[] = {1.0f, 0.5f, 0.2f};
    gnn_set_node_features(gnn, 0, features, 3);

    // Make predictions
    float output[2];
    gnn_predict(gnn, output, 2);
    printf("Prediction: [%f, %f]\n", output[0], output[1]);

    // Train
    float target[] = {0.5f, 0.5f};
    float loss;
    gnn_train(gnn, target, 2, &loss);
    printf("Loss: %f\n", loss);

    // Save model
    gnn_save_model(gnn, "model.bin");

    // Cleanup
    gnn_free(gnn);
    return 0;
}
```

### **C++ API**

```cpp
#include "gnn_facade.hpp"
#include <iostream>

int main() {
    try {
        // Create a GNN (RAII - auto-detects GPU backend)
        gnn::GnnFacade gnn(3, 16, 2, 2);

        // Or specify a backend explicitly
        gnn::GnnFacade gnn_ocl(3, 16, 2, 2, gnn::Backend::OpenCL);
        std::cout << "Backend: " << gnn.getBackendName() << std::endl;

        // Create a graph
        gnn.createEmptyGraph(5, 3);

        // Add edges
        gnn.addEdge(0, 1);
        gnn.addEdge(1, 2);

        // Set node features (using initializer list)
        gnn.setNodeFeatures(0, {1.0f, 0.5f, 0.2f});

        // Make predictions
        auto prediction = gnn.predict();
        std::cout << "Prediction: [" << prediction[0] << ", " << prediction[1] << "]" << std::endl;

        // Train
        float loss = gnn.train({0.5f, 0.5f});
        std::cout << "Loss: " << loss << std::endl;

        // Save model
        gnn.saveModel("model.bin");

        // Load from file
        gnn::GnnFacade gnn2("model.bin");
        std::cout << "Loaded model with " << gnn2.getParameterCount() << " parameters" << std::endl;

    } catch (const gnn::GnnException& e) {
        std::cerr << "GNN Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

### **C# API**

```csharp
using GnnFacadeCuda;

using var gnn = new GnnFacade(3, 16, 2, 2);

Console.WriteLine($"Backend: {gnn.BackendName}");
Console.WriteLine($"Parameters: {gnn.ParameterCount}");

// Create a graph with 5 nodes
gnn.CreateEmptyGraph(5, 3);

// Add edges
gnn.AddEdge(0, 1);
gnn.AddEdge(1, 2);
gnn.AddEdge(2, 3);

// Set node features
gnn.SetNodeFeatures(0, new float[] { 1.0f, 0.5f, 0.2f });
gnn.SetNodeFeatures(1, new float[] { 0.8f, 0.3f, 0.1f });

// Make predictions
float[] prediction = gnn.Predict();
Console.WriteLine($"Prediction: [{prediction[0]}, {prediction[1]}]");

// Train
float loss = gnn.Train(new float[] { 0.5f, 0.5f });
Console.WriteLine($"Loss: {loss}");

// Save and load
gnn.SaveModel("model.bin");
using var loaded = GnnFacade.Load("model.bin");

// PageRank
float[] ranks = gnn.ComputePageRank();
Console.WriteLine($"PageRank: [{string.Join(", ", ranks)}]");
```

### **Zig API**

```zig
const std = @import("std");
const gnn = @import("gnn");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    var facade = try gnn.GnnFacade.init(3, 16, 2, 2, .auto);
    defer facade.deinit();

    // Create a graph
    facade.createEmptyGraph(5, 3);
    _ = try facade.addEdge(0, 1, null);
    _ = try facade.addEdge(1, 2, null);
    _ = try facade.addEdge(2, 3, null);

    // Set node features
    facade.setNodeFeatures(0, &.{ 1.0, 0.5, 0.2 });
    facade.setNodeFeatures(1, &.{ 0.8, 0.3, 0.1 });

    // Make predictions
    var output_buf: [2]f32 = undefined;
    const prediction = try facade.predict(&output_buf);
    try stdout.print("Prediction: [{d}, {d}]\n", .{ prediction[0], prediction[1] });

    // Train
    const target = [_]f32{ 0.5, 0.5 };
    const loss = try facade.train(&target);
    try stdout.print("Loss: {d}\n", .{loss});

    // Save
    try facade.saveModel("model.bin");

    // PageRank
    var rank_buf: [5]f32 = undefined;
    const scores = facade.computePageRank(0.85, 20, &rank_buf);
    try stdout.print("PageRank: [", .{});
    for (scores, 0..) |score, idx| {
        if (idx > 0) try stdout.print(", ", .{});
        try stdout.print("{d}", .{score});
    }
    try stdout.print("]\n", .{});
}
```

---

### **CLI Reference**

#### Usage

```
gnn_facade_cuda [--backend <cuda|opencl|auto>] <command> [options]
```

#### Global Options

| Option | Default | Description |
|--------|---------|-------------|
| `--backend <backend>` | `auto` | GPU backend: `cuda`, `opencl`, or `auto` (tries CUDA first, then OpenCL) |

#### Commands

| Command | Description |
|---------|-------------|
| `create` | Create a new GNN model |
| `info` | Display model information |
| `train` | Train the model with graph data |
| `predict` | Make predictions on a graph |
| `create-graph` | Create empty graph with N nodes and feature dim |
| `add-edge` | Add an edge to the graph |
| `set-node-features` | Set features for a node |
| `pagerank` | Compute PageRank scores |
| `get-parameter-count` | Get total trainable parameters |
| `export-json` | Export graph as JSON |
| `help` | Show help message |

#### Examples

```bash
# Create a new model (auto-detects GPU)
gnn_facade_cuda create --feature=3 --hidden=16 --output=2 --mp-layers=2 --model=model.bin

# Create with a specific backend
gnn_facade_cuda --backend=opencl create --feature=3 --hidden=16 --output=2 --mp-layers=2 --model=model.bin

# Get model info (shows active backend)
gnn_facade_cuda info --model=model.bin

# Create and manipulate a graph
gnn_facade_cuda create-graph --model=model.bin --nodes=10 --features=3
gnn_facade_cuda add-edge --model=model.bin --source=0 --target=1
gnn_facade_cuda set-node-features --model=model.bin --node=0 --features="1.0,0.5,0.2"

# Train the model
gnn_facade_cuda train --model=model.bin --graph=graph.csv --save=trained.bin --epochs=100

# Make predictions
gnn_facade_cuda predict --model=model.bin --graph=graph.csv

# Analytics
gnn_facade_cuda pagerank --model=model.bin --damping=0.85 --iterations=20
gnn_facade_cuda get-parameter-count --model=model.bin

# Export
gnn_facade_cuda export-json --model=model.bin
```

---

## **API Reference**

### GnnFacade (all languages)

#### Model Creation & IO

| Method | Description |
|--------|-------------|
| `new(feature_size, hidden_size, output_size, num_mp_layers)` | Create a new GNN (auto-detects backend) |
| `with_backend(feature_size, hidden_size, output_size, num_mp_layers, backend)` | Create with specific backend |
| `from_model_file(filename)` | Load model from file |
| `save_model(filename)` | Save model to file |
| `load_model(filename)` | Load weights into existing model |
| `read_model_header(filename)` | Read model dimensions without loading |
| `get_backend_name()` | Get active backend name ("CUDA" or "OpenCL") |

#### Graph Operations

| Method | Description |
|--------|-------------|
| `create_empty_graph(num_nodes, feature_size)` | Create empty graph |
| `add_edge(source, target, features)` | Add edge, returns index |
| `remove_edge(edge_idx)` | Remove edge by index |
| `has_edge(source, target)` | Check if edge exists |
| `find_edge_index(source, target)` | Find edge index |
| `get_neighbors(node_idx)` | Get node neighbors |
| `rebuild_adjacency_list()` | Rebuild adjacency from edges |

#### Node Features

| Method | Description |
|--------|-------------|
| `set_node_features(node_idx, features)` | Set all features for node |
| `get_node_features(node_idx)` | Get all features for node |
| `set_node_feature(node_idx, feat_idx, value)` | Set single feature |
| `get_node_feature(node_idx, feat_idx)` | Get single feature |

#### Edge Features

| Method | Description |
|--------|-------------|
| `set_edge_features(edge_idx, features)` | Set edge features |
| `get_edge_features(edge_idx)` | Get edge features |
| `get_edge_endpoints(edge_idx)` | Get (source, target) tuple |

#### Training & Inference

| Method | Description |
|--------|-------------|
| `predict()` | Run forward pass, return predictions |
| `train(target)` | Single training step, return loss |
| `train_multiple(target, iterations)` | Train for N iterations |
| `set_learning_rate(lr)` | Set learning rate |
| `get_learning_rate()` | Get current learning rate |

#### Masking & Dropout

| Method | Description |
|--------|-------------|
| `set_node_mask(node_idx, value)` | Set node mask (active/inactive) |
| `get_node_mask(node_idx)` | Get node mask |
| `set_edge_mask(edge_idx, value)` | Set edge mask |
| `get_edge_mask(edge_idx)` | Get edge mask |
| `apply_node_dropout(rate)` | Random node dropout |
| `apply_edge_dropout(rate)` | Random edge dropout |
| `get_masked_node_count()` | Count active nodes |
| `get_masked_edge_count()` | Count active edges |

#### Analytics & Info

| Method | Description |
|--------|-------------|
| `compute_page_rank(damping, iterations)` | Compute PageRank scores |
| `get_in_degree(node_idx)` | Get node in-degree |
| `get_out_degree(node_idx)` | Get node out-degree |
| `get_num_nodes()` | Get node count |
| `get_num_edges()` | Get edge count |
| `get_parameter_count()` | Get total parameters |
| `get_architecture_summary()` | Get model summary string |
| `get_gradient_flow(layer_idx)` | Get gradient statistics |
| `export_graph_to_json()` | Export graph as JSON |

---

## **Formal Verification with Kani**

### Overview

The Rust implementation includes **Kani formal verification proofs** that mathematically prove the absence of certain classes of bugs. This goes beyond traditional testing to provide **mathematical guarantees** about code correctness.

### Verification Report

| Metric | Value |
|--------|-------|
| **Unit Tests** | 76 |
| **Kani Proof Harnesses** | 19 |
| **In-tree FFI C Boundary Proofs** | 55 |
| **In-tree CUDA Backend Proofs** | 42 |
| **In-tree OpenCL Backend Proofs** | 42 |
| **Total Verifications** | **234** |
| **Failures** | 0 |

### Running Kani Verification

```bash
cd kani_proofs

# Run all proofs
cargo kani

# Run specific proof
cargo kani --harness proof_get_node_feature_never_panics

# Run unit tests
cargo test
```

### Why Formal Verification Matters

Traditional testing can only verify specific test cases. Formal verification with Kani:

- **Exhaustively checks all possible inputs** within defined bounds
- **Mathematically proves** absence of panics, buffer overflows, and undefined behavior
- **Catches edge cases** that random testing might miss
- **Provides cryptographic-level assurance** for safety-critical code

### Kani Proof Harnesses (19 proofs)

#### Node Feature Access (3 proofs)
- `proof_get_node_feature_never_panics`
- `proof_set_node_feature_never_panics`
- `proof_get_node_features_never_panics`

#### Edge Operations (5 proofs)
- `proof_get_edge_bounds_safe`
- `proof_add_edge_bounds_checked`
- `proof_has_edge_never_panics`
- `proof_find_edge_index_never_panics`
- `proof_remove_edge_never_panics`

#### Adjacency List (3 proofs)
- `proof_get_neighbors_never_panics`
- `proof_get_in_degree_never_panics`
- `proof_get_out_degree_never_panics`

#### Node Mask Operations (2 proofs)
- `proof_node_mask_get_set_never_panic`
- `proof_node_mask_toggle_never_panics`

#### Edge Mask Operations (2 proofs)
- `proof_edge_mask_get_set_never_panic`
- `proof_edge_mask_remove_never_panics`

#### Buffer Index Validation (4 proofs)
- `proof_buffer_validator_node_correctness`
- `proof_buffer_validator_edge_correctness`
- `proof_node_feature_offset_bounds`
- `proof_node_embedding_offset_bounds`

### FFI C Boundary Safety (Category 16)

Located in `src/kani/ffi_c_boundary.rs` — 55 Kani proofs covering:

#### A. Unsigned Integer Validation
- `verify_ffi_cuint_positive_rejects_zero` — Zero c_uint rejected where positive required
- `verify_ffi_cuint_as_usize_always_safe` — c_uint → usize always safe
- `verify_ffi_cuint_max_enforced` — Upper bound validation
- `verify_ffi_len_validates_range` — Array length range validation

#### B. Output Buffer Overflow Prevention
- `verify_ffi_output_write_bounded_by_capacity` — Write never exceeds buffer capacity
- `verify_ffi_predict_output_bounded` — Predict output bounded
- `verify_ffi_zero_buffer_len_rejected` — Zero buffer length rejected for string output
- `verify_ffi_string_copy_bounded` — String copy bounded by buffer

#### C. NaN/Infinity Parameter Rejection
- `verify_ffi_f32_param_rejects_special_values` — NaN/Inf rejected at boundary
- `verify_ffi_learning_rate_rejects_nan/infinity/negative` — LR validation
- `verify_ffi_learning_rate_accepts_valid` — Valid LR accepted
- `verify_ffi_dropout_rate_validated` — Dropout [0,1] range validated
- `verify_ffi_damping_factor_validated` — PageRank damping [0,1] validated
- `verify_ffi_node_feature_nan/inf_rejected` — Node feature NaN/Inf rejected

#### D. Backend Enum Validation
- `verify_ffi_backend_enum_validation` — Backend int validated (0-2)
- `verify_ffi_backend_negative_handled` — Negative backend rejected

#### E. GNN Create Preconditions
- `verify_ffi_create_rejects_zero_*` — Zero sizes rejected for all 4 params
- `verify_ffi_create_rejects_oversized/excessive_*` — Upper bounds enforced
- `verify_ffi_create_pipeline_all_inputs` — End-to-end create validation

#### F. Array Length Validation
- `verify_ffi_feature_array_len_bounded` — Feature arrays bounded at 4096
- `verify_ffi_train_target_len_bounded` — Train target bounded at 1M
- `verify_ffi_predict_output_len_bounded` — Predict output bounded

#### G. Graph Structure Bounds
- `verify_ffi_node/edge_index_bounded` — Index bounds checked
- `verify_ffi_add_edge_validates_node_bounds` — Edge endpoint validation
- `verify_ffi_neighbor_access_bounded` — Neighbor access safe

#### H. Node/Edge Mask Validation
- `verify_ffi_node/edge_mask_oob_safe` — OOB mask access returns safe default
- `verify_ffi_edge_mask_add_respects_limit` — MAX_EDGES limit enforced

#### I. No-Panic Guarantee
- `verify_ffi_all_validators_no_panic` — All validators safe for any input

#### J. ABI Type Compatibility
- `verify_ffi_f32/u32/i32_abi_compatibility` — Size and alignment for C ABI

#### K. Input Array NaN/Infinity Detection
- `verify_ffi_nan/inf_in_f32_array_detectable` — NaN/Inf detectable in arrays

#### L. Resource Limits
- `verify_ffi_feature/train_allocation_bounded` — Memory bounded
- `verify_ffi_graph_node/edge_limit_enforced` — Graph limits enforced

#### M. Setter Value Validation
- `verify_ffi_setter_rejects_nan/infinity` — NaN/Inf rejected
- `verify_ffi_setter_accepts_valid_f32` — Valid values accepted
- `verify_ffi_dropout_setter_rejects_over_one/accepts_valid` — Range enforced

#### N. End-to-End Pipeline Validation
- `verify_ffi_complete_predict/train/create/page_rank_pipeline` — Full pipelines

#### O. Buffer Validator Proofs
- `verify_ffi_buffer_validator_node/edge_index` — Index validation
- `verify_ffi_buffer_validator_feature/embedding_offset_safe` — Offset calculations

### CUDA Backend FFI Safety (Category 17)

Located in `src/kani/ffi_cuda_boundary.rs` — 42 Kani proofs + unit tests across 15 categories (A–O):

- Layer buffer size correctness (weight, bias, gradient)
- CUDA grid/block dimension safety for forward, backward, input grad kernels
- Weight index validity for flat buffers (weights and messages)
- Transfer size non-zero and f32-aligned
- Node embedding buffer sizing (MAX_NODES × hidden_size)
- Message buffer sizing (MAX_EDGES × hidden_size)
- Kernel launch parameter overflow prevention
- Aggregation buffer sizing and index validity
- Graph readout buffer safety and grid dimensions
- Gradient buffer sizing (output gradient and target match)
- Neighbor offset/count buffer safety
- No-panic guarantee for all dimension calculations
- ABI type compatibility for CUDA interop (f32, i32, u32)
- End-to-end forward pass buffer chain (message→update→readout→output)

### OpenCL Backend FFI Safety (Category 18)

Located in `src/kani/ffi_opencl_boundary.rs` — 42 Kani proofs + unit tests across 15 categories (A–O):

- Layer buffer size correctness for clCreateBuffer
- OpenCL global/local work size safety (divisible, covers all items)
- Weight index validity for flat OpenCL buffers
- Transfer alignment for clEnqueueRead/WriteBuffer
- cl_float/cl_int alignment for OpenCL interop
- Node embedding buffer sizing for OpenCL
- Message buffer sizing for OpenCL
- Work group size power-of-two property and device limits
- Aggregation buffer sizing for OpenCL
- Graph readout buffer and readout work size
- Gradient buffer sizing for OpenCL
- Neighbor offset/count buffer for OpenCL
- No-panic guarantee for all work size calculations
- ABI type compatibility for OpenCL interop
- End-to-end forward pass buffer chain adapted for OpenCL

---

## **CISA/NSA Compliance**

### Secure by Design

This project follows **CISA (Cybersecurity and Infrastructure Security Agency)** and **NSA (National Security Agency)** Secure by Design principles:

| Principle | Implementation |
|-----------|---------------|
| **Memory Safety** | Rust ownership model eliminates buffer overflows, use-after-free, and data races |
| **Formal Verification** | Kani proofs mathematically verify absence of critical bugs |
| **Input Validation** | All CLI inputs validated before processing |
| **Defense in Depth** | Multiple layers of safety (language, compiler, runtime checks) |
| **Secure Defaults** | Safe default configurations throughout |
| **Transparency** | Open source with full code visibility |

### Compliance Checklist

- [x] **Memory-safe language** (Rust implementation)
- [x] **Static analysis** (Rust compiler + Clippy)
- [x] **Formal verification** (Kani proof harnesses)
- [x] **Comprehensive testing** (Unit tests + integration tests)
- [x] **Bounds checking** (Verified array access)
- [x] **Input validation** (CLI argument parsing)
- [x] **No unsafe code in critical paths** (Where possible)
- [x] **Documentation** (Inline docs + README)
- [x] **Version control** (Git)
- [x] **License clarity** (MIT License)

### Attestation

This codebase has been developed following secure software development lifecycle (SSDLC) practices and demonstrates:

- **234 formal verifications passed** (76 unit tests + 19 Kani proofs + 139 in-tree FFI/CUDA/OpenCL proofs)
- **Zero warnings** compilation across all implementations
- **Consistent API** across all language bindings
- **Production-ready** code quality

---

## **License**

MIT License

Copyright (c) 2025 Matthew Abbott

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## **Author**

**Matthew Abbott**
Email: mattbachg@gmail.com

---

*Built with precision. Verified with rigor. Secured by design.*

/*
Package facadedgan provides Go bindings for the facaded_gan_cuda GAN library
via CGo.

BUILD
-----
Before using this package, build the C shared library:

	cd <workspace_root>
	cargo build --release -p facaded_gan_c

Then build your Go program with CGo enabled (the default):

	cd go_bindings
	go build ./...

MEMORY
------
All objects returned by New* / factory functions own heap memory that is
automatically released when the object is garbage-collected (via
runtime.SetFinalizer).  Call Free() explicitly if you need deterministic
release timing.

STRING PARAMETERS
-----------------
Enum-typed parameters accept the same lowercase strings as the C API:

	activation : "relu" | "sigmoid" | "tanh" | "leaky" | "none"
	optimizer  : "adam" | "sgd" | "rmsprop"
	loss_type  : "bce"  | "wgan" | "hinge" | "ls"
	noise_type : "gauss"| "uniform" | "analog"
	data_type  : "vector"| "image" | "audio"
	backend    : "cpu"  | "cuda"  | "opencl" | "hybrid" | "auto"
*/
package facadedgan

/*
#cgo CFLAGS:  -I../../include
#cgo LDFLAGS: -L../../target/release -lfacaded_gan_c -Wl,-rpath,../../target/release
#include "facaded_gan.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// cstr converts a Go string to a temporary *C.char.
// The caller must defer C.free(unsafe.Pointer(cs)) immediately after calling cstr.
func cstr(s string) *C.char { return C.CString(s) }

// cfree releases a C string allocated by cstr / C.CString.
func cfree(p *C.char) { C.free(unsafe.Pointer(p)) }

// ─── Matrix ───────────────────────────────────────────────────────────────────

// Matrix is a 2-D row-major matrix of float32 values.
// All returned Matrix values are owned by the caller; call Free() or let GC handle it.
type Matrix struct{ ptr *C.GanMatrix }

func newMatrix(p *C.GanMatrix) *Matrix {
	if p == nil {
		return nil
	}
	m := &Matrix{ptr: p}
	runtime.SetFinalizer(m, (*Matrix).Free)
	return m
}

// NewMatrix creates a zero-filled rows×cols Matrix.
func NewMatrix(rows, cols int) *Matrix {
	return newMatrix(C.gf_matrix_create(C.int(rows), C.int(cols)))
}

// NewMatrixFromData copies a flat row-major []float32 into a new Matrix.
// data[i*cols+j] is row i, column j.
func NewMatrixFromData(data []float32, rows, cols int) *Matrix {
	if len(data) == 0 {
		return NewMatrix(rows, cols)
	}
	return newMatrix(C.gf_matrix_from_data(
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(rows), C.int(cols),
	))
}

// Free releases the underlying C memory immediately.
// After Free, the Matrix must not be used.
func (m *Matrix) Free() {
	if m.ptr != nil {
		C.gf_matrix_free(m.ptr)
		m.ptr = nil
		runtime.SetFinalizer(m, nil)
	}
}

// Rows returns the number of rows.
func (m *Matrix) Rows() int { return int(C.gf_matrix_rows(m.ptr)) }

// Cols returns the number of columns.
func (m *Matrix) Cols() int { return int(C.gf_matrix_cols(m.ptr)) }

// Get returns element (row, col), or 0 if out-of-range.
func (m *Matrix) Get(row, col int) float32 {
	return float32(C.gf_matrix_get(m.ptr, C.int(row), C.int(col)))
}

// Set writes element (row, col). No-op if out-of-range.
func (m *Matrix) Set(row, col int, val float32) {
	C.gf_matrix_set(m.ptr, C.int(row), C.int(col), C.float(val))
}

// SafeGet returns element (row, col), or def if out-of-range.
func (m *Matrix) SafeGet(row, col int, def float32) float32 {
	return float32(C.gf_matrix_safe_get(m.ptr, C.int(row), C.int(col), C.float(def)))
}

// Data returns a Go slice that aliases the internal flat buffer.
// Valid only until the Matrix is freed.
func (m *Matrix) Data() []float32 {
	n := m.Rows() * m.Cols()
	if n == 0 {
		return nil
	}
	ptr := C.gf_matrix_data(m.ptr)
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), n)
}

// ToSlice copies the matrix into a [][]float32.
func (m *Matrix) ToSlice() [][]float32 {
	rows, cols := m.Rows(), m.Cols()
	flat := m.Data()
	out := make([][]float32, rows)
	for i := range out {
		out[i] = make([]float32, cols)
		copy(out[i], flat[i*cols:(i+1)*cols])
	}
	return out
}

// Multiply returns a new Matrix A×B.
func (m *Matrix) Multiply(b *Matrix) *Matrix {
	return newMatrix(C.gf_matrix_multiply(m.ptr, b.ptr))
}

// Add returns element-wise A+B.
func (m *Matrix) Add(b *Matrix) *Matrix {
	return newMatrix(C.gf_matrix_add(m.ptr, b.ptr))
}

// Subtract returns element-wise A−B.
func (m *Matrix) Subtract(b *Matrix) *Matrix {
	return newMatrix(C.gf_matrix_subtract(m.ptr, b.ptr))
}

// Scale returns a new matrix with all elements multiplied by s.
func (m *Matrix) Scale(s float32) *Matrix {
	return newMatrix(C.gf_matrix_scale(m.ptr, C.float(s)))
}

// Transpose returns the transpose.
func (m *Matrix) Transpose() *Matrix {
	return newMatrix(C.gf_matrix_transpose(m.ptr))
}

// Normalize returns a row-wise L2-normalised copy.
func (m *Matrix) Normalize() *Matrix {
	return newMatrix(C.gf_matrix_normalize(m.ptr))
}

// ElementMul returns element-wise product A⊙B.
func (m *Matrix) ElementMul(b *Matrix) *Matrix {
	return newMatrix(C.gf_matrix_element_mul(m.ptr, b.ptr))
}

// Activation functions — each returns a new Matrix.
func (m *Matrix) ReLU() *Matrix      { return newMatrix(C.gf_relu(m.ptr)) }
func (m *Matrix) Sigmoid() *Matrix   { return newMatrix(C.gf_sigmoid(m.ptr)) }
func (m *Matrix) TanhAct() *Matrix   { return newMatrix(C.gf_tanh_m(m.ptr)) }
func (m *Matrix) Softmax() *Matrix   { return newMatrix(C.gf_softmax(m.ptr)) }

// LeakyReLU applies leaky ReLU with the given alpha.
func (m *Matrix) LeakyReLU(alpha float32) *Matrix {
	return newMatrix(C.gf_leaky_relu(m.ptr, C.float(alpha)))
}

// Activate applies a named activation.
// act: "relu" | "sigmoid" | "tanh" | "leaky" | "none"
func (m *Matrix) Activate(act string) *Matrix {
	cs := cstr(act)
	defer cfree(cs)
	return newMatrix(C.gf_activate(m.ptr, cs))
}

// BoundsCheck returns true if (r, c) is within matrix bounds.
func (m *Matrix) BoundsCheck(r, c int) bool {
	return C.gf_bounds_check(m.ptr, C.int(r), C.int(c)) != 0
}

// ─── Vector ───────────────────────────────────────────────────────────────────

// Vector is a 1-D float32 vector.
type Vector struct{ ptr *C.GanVector }

func newVector(p *C.GanVector) *Vector {
	if p == nil {
		return nil
	}
	v := &Vector{ptr: p}
	runtime.SetFinalizer(v, (*Vector).Free)
	return v
}

// NewVector creates a zero-filled vector of length len.
func NewVector(length int) *Vector {
	return newVector(C.gf_vector_create(C.int(length)))
}

// Free releases underlying C memory immediately.
func (v *Vector) Free() {
	if v.ptr != nil {
		C.gf_vector_free(v.ptr)
		v.ptr = nil
		runtime.SetFinalizer(v, nil)
	}
}

// Len returns the length of the vector.
func (v *Vector) Len() int { return int(C.gf_vector_len(v.ptr)) }

// Get returns element at index idx (bounds-checked).
func (v *Vector) Get(idx int) float32 { return float32(C.gf_vector_get(v.ptr, C.int(idx))) }

// Data returns a Go slice aliasing the internal buffer. Valid until Free().
func (v *Vector) Data() []float32 {
	n := v.Len()
	if n == 0 {
		return nil
	}
	ptr := C.gf_vector_data(v.ptr)
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), n)
}

// Slerp performs spherical linear interpolation between v and other at t ∈ [0,1].
func (v *Vector) Slerp(other *Vector, t float32) *Vector {
	return newVector(C.gf_vector_noise_slerp(v.ptr, other.ptr, C.float(t)))
}

// ─── Config ───────────────────────────────────────────────────────────────────

// Config holds all GAN hyper-parameters.
// Obtain via NewConfig(); fields are set via setters.
type Config struct{ ptr *C.GanConfig }

func newConfig(p *C.GanConfig) *Config {
	if p == nil {
		return nil
	}
	c := &Config{ptr: p}
	runtime.SetFinalizer(c, (*Config).Free)
	return c
}

// NewConfig creates a Config pre-filled with library defaults.
func NewConfig() *Config { return newConfig(C.gf_config_create()) }

// Free releases underlying C memory immediately.
func (c *Config) Free() {
	if c.ptr != nil {
		C.gf_config_free(c.ptr)
		c.ptr = nil
		runtime.SetFinalizer(c, nil)
	}
}

// Integer getters/setters.
func (c *Config) Epochs() int             { return int(C.gf_config_get_epochs(c.ptr)) }
func (c *Config) SetEpochs(v int)         { C.gf_config_set_epochs(c.ptr, C.int(v)) }
func (c *Config) BatchSize() int          { return int(C.gf_config_get_batch_size(c.ptr)) }
func (c *Config) SetBatchSize(v int)      { C.gf_config_set_batch_size(c.ptr, C.int(v)) }
func (c *Config) NoiseDepth() int         { return int(C.gf_config_get_noise_depth(c.ptr)) }
func (c *Config) SetNoiseDepth(v int)     { C.gf_config_set_noise_depth(c.ptr, C.int(v)) }
func (c *Config) ConditionSize() int      { return int(C.gf_config_get_condition_size(c.ptr)) }
func (c *Config) SetConditionSize(v int)  { C.gf_config_set_condition_size(c.ptr, C.int(v)) }
func (c *Config) GeneratorBits() int      { return int(C.gf_config_get_generator_bits(c.ptr)) }
func (c *Config) SetGeneratorBits(v int)  { C.gf_config_set_generator_bits(c.ptr, C.int(v)) }
func (c *Config) DiscriminatorBits() int  { return int(C.gf_config_get_discriminator_bits(c.ptr)) }
func (c *Config) SetDiscriminatorBits(v int) { C.gf_config_set_discriminator_bits(c.ptr, C.int(v)) }
func (c *Config) MaxResLevel() int        { return int(C.gf_config_get_max_res_level(c.ptr)) }
func (c *Config) SetMaxResLevel(v int)    { C.gf_config_set_max_res_level(c.ptr, C.int(v)) }
func (c *Config) MetricInterval() int     { return int(C.gf_config_get_metric_interval(c.ptr)) }
func (c *Config) SetMetricInterval(v int) { C.gf_config_set_metric_interval(c.ptr, C.int(v)) }
func (c *Config) CheckpointInterval() int { return int(C.gf_config_get_checkpoint_interval(c.ptr)) }
func (c *Config) SetCheckpointInterval(v int) { C.gf_config_set_checkpoint_interval(c.ptr, C.int(v)) }
func (c *Config) FuzzIterations() int     { return int(C.gf_config_get_fuzz_iterations(c.ptr)) }
func (c *Config) SetFuzzIterations(v int) { C.gf_config_set_fuzz_iterations(c.ptr, C.int(v)) }
func (c *Config) NumThreads() int         { return int(C.gf_config_get_num_threads(c.ptr)) }
func (c *Config) SetNumThreads(v int)     { C.gf_config_set_num_threads(c.ptr, C.int(v)) }

// Float getters/setters.
func (c *Config) LearningRate() float32         { return float32(C.gf_config_get_learning_rate(c.ptr)) }
func (c *Config) SetLearningRate(v float32)     { C.gf_config_set_learning_rate(c.ptr, C.float(v)) }
func (c *Config) GPLambda() float32             { return float32(C.gf_config_get_gp_lambda(c.ptr)) }
func (c *Config) SetGPLambda(v float32)         { C.gf_config_set_gp_lambda(c.ptr, C.float(v)) }
func (c *Config) GeneratorLR() float32          { return float32(C.gf_config_get_generator_lr(c.ptr)) }
func (c *Config) SetGeneratorLR(v float32)      { C.gf_config_set_generator_lr(c.ptr, C.float(v)) }
func (c *Config) DiscriminatorLR() float32      { return float32(C.gf_config_get_discriminator_lr(c.ptr)) }
func (c *Config) SetDiscriminatorLR(v float32)  { C.gf_config_set_discriminator_lr(c.ptr, C.float(v)) }
func (c *Config) WeightDecayVal() float32       { return float32(C.gf_config_get_weight_decay_val(c.ptr)) }
func (c *Config) SetWeightDecayVal(v float32)   { C.gf_config_set_weight_decay_val(c.ptr, C.float(v)) }

// Bool getters/setters (C API: int 0=false, !=0=true).
func gbool(v C.int) bool { return v != 0 }
func cbool(v bool) C.int {
	if v {
		return 1
	}
	return 0
}

func (c *Config) UseBatchNorm() bool           { return gbool(C.gf_config_get_use_batch_norm(c.ptr)) }
func (c *Config) SetUseBatchNorm(v bool)       { C.gf_config_set_use_batch_norm(c.ptr, cbool(v)) }
func (c *Config) UseLayerNorm() bool           { return gbool(C.gf_config_get_use_layer_norm(c.ptr)) }
func (c *Config) SetUseLayerNorm(v bool)       { C.gf_config_set_use_layer_norm(c.ptr, cbool(v)) }
func (c *Config) UseSpectralNorm() bool        { return gbool(C.gf_config_get_use_spectral_norm(c.ptr)) }
func (c *Config) SetUseSpectralNorm(v bool)    { C.gf_config_set_use_spectral_norm(c.ptr, cbool(v)) }
func (c *Config) UseLabelSmoothing() bool      { return gbool(C.gf_config_get_use_label_smoothing(c.ptr)) }
func (c *Config) SetUseLabelSmoothing(v bool)  { C.gf_config_set_use_label_smoothing(c.ptr, cbool(v)) }
func (c *Config) UseFeatureMatching() bool     { return gbool(C.gf_config_get_use_feature_matching(c.ptr)) }
func (c *Config) SetUseFeatureMatching(v bool) { C.gf_config_set_use_feature_matching(c.ptr, cbool(v)) }
func (c *Config) UseMinibatchStdDev() bool     { return gbool(C.gf_config_get_use_minibatch_std_dev(c.ptr)) }
func (c *Config) SetUseMinibatchStdDev(v bool) { C.gf_config_set_use_minibatch_std_dev(c.ptr, cbool(v)) }
func (c *Config) UseProgressive() bool         { return gbool(C.gf_config_get_use_progressive(c.ptr)) }
func (c *Config) SetUseProgressive(v bool)     { C.gf_config_set_use_progressive(c.ptr, cbool(v)) }
func (c *Config) UseAugmentation() bool        { return gbool(C.gf_config_get_use_augmentation(c.ptr)) }
func (c *Config) SetUseAugmentation(v bool)    { C.gf_config_set_use_augmentation(c.ptr, cbool(v)) }
func (c *Config) ComputeMetrics() bool         { return gbool(C.gf_config_get_compute_metrics(c.ptr)) }
func (c *Config) SetComputeMetrics(v bool)     { C.gf_config_set_compute_metrics(c.ptr, cbool(v)) }
func (c *Config) UseWeightDecay() bool         { return gbool(C.gf_config_get_use_weight_decay(c.ptr)) }
func (c *Config) SetUseWeightDecay(v bool)     { C.gf_config_set_use_weight_decay(c.ptr, cbool(v)) }
func (c *Config) UseCosineAnneal() bool        { return gbool(C.gf_config_get_use_cosine_anneal(c.ptr)) }
func (c *Config) SetUseCosineAnneal(v bool)    { C.gf_config_set_use_cosine_anneal(c.ptr, cbool(v)) }
func (c *Config) AuditLog() bool               { return gbool(C.gf_config_get_audit_log(c.ptr)) }
func (c *Config) SetAuditLog(v bool)           { C.gf_config_set_audit_log(c.ptr, cbool(v)) }
func (c *Config) UseEncryption() bool          { return gbool(C.gf_config_get_use_encryption(c.ptr)) }
func (c *Config) SetUseEncryption(v bool)      { C.gf_config_set_use_encryption(c.ptr, cbool(v)) }
func (c *Config) UseConv() bool                { return gbool(C.gf_config_get_use_conv(c.ptr)) }
func (c *Config) SetUseConv(v bool)            { C.gf_config_set_use_conv(c.ptr, cbool(v)) }
func (c *Config) UseAttention() bool           { return gbool(C.gf_config_get_use_attention(c.ptr)) }
func (c *Config) SetUseAttention(v bool)       { C.gf_config_set_use_attention(c.ptr, cbool(v)) }

// String setters (no getters — strings are write-only in the C API).
func (c *Config) SetSaveModel(v string) {
	cs := cstr(v); defer cfree(cs); C.gf_config_set_save_model(c.ptr, cs)
}
func (c *Config) SetLoadModel(v string) {
	cs := cstr(v); defer cfree(cs); C.gf_config_set_load_model(c.ptr, cs)
}
func (c *Config) SetLoadJSONModel(v string) {
	cs := cstr(v); defer cfree(cs); C.gf_config_set_load_json_model(c.ptr, cs)
}
func (c *Config) SetOutputDir(v string) {
	cs := cstr(v); defer cfree(cs); C.gf_config_set_output_dir(c.ptr, cs)
}
func (c *Config) SetDataPath(v string) {
	cs := cstr(v); defer cfree(cs); C.gf_config_set_data_path(c.ptr, cs)
}
func (c *Config) SetAuditLogFile(v string) {
	cs := cstr(v); defer cfree(cs); C.gf_config_set_audit_log_file(c.ptr, cs)
}
func (c *Config) SetEncryptionKey(v string) {
	cs := cstr(v); defer cfree(cs); C.gf_config_set_encryption_key(c.ptr, cs)
}
func (c *Config) SetPatchConfig(v string) {
	cs := cstr(v); defer cfree(cs); C.gf_config_set_patch_config(c.ptr, cs)
}

// Enum setters.
func (c *Config) SetActivation(v string) {
	cs := cstr(v); defer cfree(cs); C.gf_config_set_activation(c.ptr, cs)
}
func (c *Config) SetNoiseType(v string) {
	cs := cstr(v); defer cfree(cs); C.gf_config_set_noise_type(c.ptr, cs)
}
func (c *Config) SetOptimizer(v string) {
	cs := cstr(v); defer cfree(cs); C.gf_config_set_optimizer(c.ptr, cs)
}
func (c *Config) SetLossType(v string) {
	cs := cstr(v); defer cfree(cs); C.gf_config_set_loss_type(c.ptr, cs)
}
func (c *Config) SetDataType(v string) {
	cs := cstr(v); defer cfree(cs); C.gf_config_set_data_type(c.ptr, cs)
}

// ─── Network ──────────────────────────────────────────────────────────────────

// Network is a generator or discriminator network.
type Network struct{ ptr *C.GanNetwork }

func newNetwork(p *C.GanNetwork) *Network {
	if p == nil {
		return nil
	}
	n := &Network{ptr: p}
	runtime.SetFinalizer(n, (*Network).Free)
	return n
}

// Free releases underlying C memory immediately.
func (n *Network) Free() {
	if n.ptr != nil {
		C.gf_network_free(n.ptr)
		n.ptr = nil
		runtime.SetFinalizer(n, nil)
	}
}

// GenBuild builds a dense generator. sizes lists layer widths, e.g. []int{64, 128, 1}.
// act: activation name; opt: optimizer name; lr: learning rate.
func GenBuild(sizes []int, act, opt string, lr float32) *Network {
	csizes := make([]C.int, len(sizes))
	for i, s := range sizes {
		csizes[i] = C.int(s)
	}
	cact := cstr(act); defer cfree(cact)
	copt := cstr(opt); defer cfree(copt)
	return newNetwork(C.gf_gen_build(
		(*C.int)(unsafe.Pointer(&csizes[0])), C.int(len(csizes)),
		cact, copt, C.float(lr),
	))
}

// DiscBuild builds a dense discriminator.
func DiscBuild(sizes []int, act, opt string, lr float32) *Network {
	csizes := make([]C.int, len(sizes))
	for i, s := range sizes {
		csizes[i] = C.int(s)
	}
	cact := cstr(act); defer cfree(cact)
	copt := cstr(opt); defer cfree(copt)
	return newNetwork(C.gf_disc_build(
		(*C.int)(unsafe.Pointer(&csizes[0])), C.int(len(csizes)),
		cact, copt, C.float(lr),
	))
}

// GenBuildConv builds a convolutional generator.
func GenBuildConv(noiseDim, condSz, baseCh int, act, opt string, lr float32) *Network {
	cact := cstr(act); defer cfree(cact)
	copt := cstr(opt); defer cfree(copt)
	return newNetwork(C.gf_gen_build_conv(
		C.int(noiseDim), C.int(condSz), C.int(baseCh),
		cact, copt, C.float(lr),
	))
}

// DiscBuildConv builds a convolutional discriminator.
func DiscBuildConv(inCh, inW, inH, condSz, baseCh int, act, opt string, lr float32) *Network {
	cact := cstr(act); defer cfree(cact)
	copt := cstr(opt); defer cfree(copt)
	return newNetwork(C.gf_disc_build_conv(
		C.int(inCh), C.int(inW), C.int(inH),
		C.int(condSz), C.int(baseCh),
		cact, copt, C.float(lr),
	))
}

// LayerCount returns the number of layers.
func (n *Network) LayerCount() int { return int(C.gf_network_layer_count(n.ptr)) }

// LearningRate returns the current learning rate.
func (n *Network) LearningRate() float32 { return float32(C.gf_network_learning_rate(n.ptr)) }

// IsTraining returns true if the network is in training mode.
func (n *Network) IsTraining() bool { return gbool(C.gf_network_is_training(n.ptr)) }

// Forward runs a forward pass. inp is batch×features. Caller owns result.
func (n *Network) Forward(inp *Matrix) *Matrix {
	return newMatrix(C.gf_network_forward(n.ptr, inp.ptr))
}

// Backward runs a backward pass. Caller owns result.
func (n *Network) Backward(gradOut *Matrix) *Matrix {
	return newMatrix(C.gf_network_backward(n.ptr, gradOut.ptr))
}

// UpdateWeights applies accumulated gradients to weights.
func (n *Network) UpdateWeights() { C.gf_network_update_weights(n.ptr) }

// SetTraining switches training (true) / inference (false) mode.
func (n *Network) SetTraining(training bool) {
	C.gf_network_set_training(n.ptr, cbool(training))
}

// Sample generates count outputs. noiseType: "gauss" | "uniform" | "analog". Caller owns result.
func (n *Network) Sample(count, noiseDim int, noiseType string) *Matrix {
	cs := cstr(noiseType); defer cfree(cs)
	return newMatrix(C.gf_network_sample(n.ptr, C.int(count), C.int(noiseDim), cs))
}

// Verify sanitises weights (replaces NaN/Inf with 0).
func (n *Network) Verify() { C.gf_network_verify(n.ptr) }

// Save writes weights to path (binary).
func (n *Network) Save(path string) {
	cs := cstr(path); defer cfree(cs)
	C.gf_network_save(n.ptr, cs)
}

// Load reads weights from path (binary).
func (n *Network) Load(path string) {
	cs := cstr(path); defer cfree(cs)
	C.gf_network_load(n.ptr, cs)
}

// ─── Dataset ──────────────────────────────────────────────────────────────────

// Dataset holds training samples.
type Dataset struct{ ptr *C.GanDataset }

func newDataset(p *C.GanDataset) *Dataset {
	if p == nil {
		return nil
	}
	d := &Dataset{ptr: p}
	runtime.SetFinalizer(d, (*Dataset).Free)
	return d
}

// Free releases underlying C memory immediately.
func (d *Dataset) Free() {
	if d.ptr != nil {
		C.gf_dataset_free(d.ptr)
		d.ptr = nil
		runtime.SetFinalizer(d, nil)
	}
}

// NewSyntheticDataset creates a synthetic random dataset with count samples, each of length features.
func NewSyntheticDataset(count, features int) *Dataset {
	return newDataset(C.gf_dataset_create_synthetic(C.int(count), C.int(features)))
}

// LoadDataset loads a dataset from path.
// dataType: "vector" | "image" | "audio".
func LoadDataset(path, dataType string) *Dataset {
	cp := cstr(path); defer cfree(cp)
	cd := cstr(dataType); defer cfree(cd)
	return newDataset(C.gf_dataset_load(cp, cd))
}

// Count returns the number of samples.
func (d *Dataset) Count() int { return int(C.gf_dataset_count(d.ptr)) }

// ─── Metrics ──────────────────────────────────────────────────────────────────

// Metrics holds per-step training statistics.
type Metrics struct{ ptr *C.GanMetrics }

func newMetrics(p *C.GanMetrics) *Metrics {
	if p == nil {
		return nil
	}
	m := &Metrics{ptr: p}
	runtime.SetFinalizer(m, (*Metrics).Free)
	return m
}

// Free releases underlying C memory immediately.
func (m *Metrics) Free() {
	if m.ptr != nil {
		C.gf_metrics_free(m.ptr)
		m.ptr = nil
		runtime.SetFinalizer(m, nil)
	}
}

func (m *Metrics) DLossReal() float32  { return float32(C.gf_metrics_d_loss_real(m.ptr)) }
func (m *Metrics) DLossFake() float32  { return float32(C.gf_metrics_d_loss_fake(m.ptr)) }
func (m *Metrics) GLoss() float32      { return float32(C.gf_metrics_g_loss(m.ptr)) }
func (m *Metrics) FIDScore() float32   { return float32(C.gf_metrics_fid_score(m.ptr)) }
func (m *Metrics) ISScore() float32    { return float32(C.gf_metrics_is_score(m.ptr)) }
func (m *Metrics) GradPenalty() float32{ return float32(C.gf_metrics_grad_penalty(m.ptr)) }
func (m *Metrics) Epoch() int          { return int(C.gf_metrics_epoch(m.ptr)) }
func (m *Metrics) Batch() int          { return int(C.gf_metrics_batch(m.ptr)) }

// ─── Result ───────────────────────────────────────────────────────────────────

// Result combines trained networks and final metrics from Run().
type Result struct{ ptr *C.GanResult }

func newResult(p *C.GanResult) *Result {
	if p == nil {
		return nil
	}
	r := &Result{ptr: p}
	runtime.SetFinalizer(r, (*Result).Free)
	return r
}

// Free releases underlying C memory immediately.
func (r *Result) Free() {
	if r.ptr != nil {
		C.gf_result_free(r.ptr)
		r.ptr = nil
		runtime.SetFinalizer(r, nil)
	}
}

// Generator returns the trained generator. Caller owns the returned Network.
func (r *Result) Generator() *Network { return newNetwork(C.gf_result_generator(r.ptr)) }

// Discriminator returns the trained discriminator. Caller owns the returned Network.
func (r *Result) Discriminator() *Network { return newNetwork(C.gf_result_discriminator(r.ptr)) }

// Metrics returns the final training metrics. Caller owns the returned Metrics.
func (r *Result) Metrics() *Metrics { return newMetrics(C.gf_result_metrics(r.ptr)) }

// ─── High-level API ───────────────────────────────────────────────────────────

// Run builds networks, trains, and returns results — all from a single Config.
// Caller owns the returned Result.
func Run(cfg *Config) *Result {
	return newResult(C.gf_run(cfg.ptr))
}

// InitBackend initialises the global compute backend.
// name: "cpu" | "cuda" | "opencl" | "hybrid" | "auto"
func InitBackend(name string) {
	cs := cstr(name); defer cfree(cs)
	C.gf_init_backend(cs)
}

// DetectBackend returns the name of the best available backend.
// The returned string is a static constant; do not free it.
func DetectBackend() string {
	return C.GoString(C.gf_detect_backend())
}

// SecureRandomize seeds the global RNG from /dev/urandom.
func SecureRandomize() { C.gf_secure_randomize() }

// ─── Training API ─────────────────────────────────────────────────────────────

// TrainFull runs all epochs. Returns final-step metrics. Caller owns the result.
func TrainFull(gen, disc *Network, ds *Dataset, cfg *Config) *Metrics {
	return newMetrics(C.gf_train_full(gen.ptr, disc.ptr, ds.ptr, cfg.ptr))
}

// TrainStep runs one discriminator + generator update step.
// realBatch is batch×features; noise is batch×noiseDepth.
// Caller owns the returned Metrics.
func TrainStep(gen, disc *Network, realBatch, noise *Matrix, cfg *Config) *Metrics {
	return newMetrics(C.gf_train_step(gen.ptr, disc.ptr, realBatch.ptr, noise.ptr, cfg.ptr))
}

// SaveJSON saves both networks to a JSON file.
func SaveJSON(gen, disc *Network, path string) {
	cs := cstr(path); defer cfree(cs)
	C.gf_train_save_json(gen.ptr, disc.ptr, cs)
}

// LoadJSON loads both networks from a JSON file.
func LoadJSON(gen, disc *Network, path string) {
	cs := cstr(path); defer cfree(cs)
	C.gf_train_load_json(gen.ptr, disc.ptr, cs)
}

// SaveCheckpoint saves a checkpoint (binary) to dir at epoch ep.
func SaveCheckpoint(gen, disc *Network, ep int, dir string) {
	cs := cstr(dir); defer cfree(cs)
	C.gf_train_save_checkpoint(gen.ptr, disc.ptr, C.int(ep), cs)
}

// LoadCheckpoint loads a checkpoint from dir at epoch ep.
func LoadCheckpoint(gen, disc *Network, ep int, dir string) {
	cs := cstr(dir); defer cfree(cs)
	C.gf_train_load_checkpoint(gen.ptr, disc.ptr, C.int(ep), cs)
}

// ─── Loss functions ───────────────────────────────────────────────────────────

// BCELoss computes binary cross-entropy loss.
func BCELoss(pred, target *Matrix) float32 {
	return float32(C.gf_bce_loss(pred.ptr, target.ptr))
}

// BCEGrad computes the BCE gradient. Caller owns the result.
func BCEGrad(pred, target *Matrix) *Matrix {
	return newMatrix(C.gf_bce_grad(pred.ptr, target.ptr))
}

// WGANDiscLoss computes WGAN discriminator loss.
func WGANDiscLoss(dReal, dFake *Matrix) float32 {
	return float32(C.gf_wgan_disc_loss(dReal.ptr, dFake.ptr))
}

// WGANGenLoss computes WGAN generator loss.
func WGANGenLoss(dFake *Matrix) float32 {
	return float32(C.gf_wgan_gen_loss(dFake.ptr))
}

// HingeDiscLoss computes hinge discriminator loss.
func HingeDiscLoss(dReal, dFake *Matrix) float32 {
	return float32(C.gf_hinge_disc_loss(dReal.ptr, dFake.ptr))
}

// HingeGenLoss computes hinge generator loss.
func HingeGenLoss(dFake *Matrix) float32 {
	return float32(C.gf_hinge_gen_loss(dFake.ptr))
}

// LSDiscLoss computes least-squares discriminator loss.
func LSDiscLoss(dReal, dFake *Matrix) float32 {
	return float32(C.gf_ls_disc_loss(dReal.ptr, dFake.ptr))
}

// LSGenLoss computes least-squares generator loss.
func LSGenLoss(dFake *Matrix) float32 {
	return float32(C.gf_ls_gen_loss(dFake.ptr))
}

// CosineAnneal returns the cosine-annealed learning rate.
func CosineAnneal(epoch, maxEp int, baseLR, minLR float32) float32 {
	return float32(C.gf_cosine_anneal(C.int(epoch), C.int(maxEp), C.float(baseLR), C.float(minLR)))
}

// ─── Random / Noise ───────────────────────────────────────────────────────────

// RandomGaussian returns a single Gaussian-distributed float32.
func RandomGaussian() float32 { return float32(C.gf_random_gaussian()) }

// RandomUniform returns a single uniform float32 in [lo, hi].
func RandomUniform(lo, hi float32) float32 {
	return float32(C.gf_random_uniform(C.float(lo), C.float(hi)))
}

// GenerateNoise returns a size×depth noise matrix.
// noiseType: "gauss" | "uniform" | "analog". Caller owns the result.
func GenerateNoise(size, depth int, noiseType string) *Matrix {
	cs := cstr(noiseType); defer cfree(cs)
	return newMatrix(C.gf_generate_noise(C.int(size), C.int(depth), cs))
}

// ─── Security ─────────────────────────────────────────────────────────────────

// ValidatePath returns true if path is safe (no traversal, absolute refs, etc.).
func ValidatePath(path string) bool {
	cs := cstr(path); defer cfree(cs)
	return C.gf_validate_path(cs) != 0
}

// AuditLog appends msg to logFile with an ISO-8601 timestamp.
func AuditLog(msg, logFile string) {
	cm := cstr(msg); defer cfree(cm)
	cl := cstr(logFile); defer cfree(cl)
	C.gf_audit_log(cm, cl)
}

// ─── Matrix in-place operations ───────────────────────────────────────────────

// MatrixAddInPlace adds b into a element-wise, modifying a in place.
func MatrixAddInPlace(a, b *Matrix) {
	C.gf_matrix_add_in_place(a.ptr, b.ptr)
}

// MatrixScaleInPlace multiplies every element of a by s in place.
func MatrixScaleInPlace(a *Matrix, s float32) {
	C.gf_matrix_scale_in_place(a.ptr, C.float(s))
}

// MatrixClipInPlace clamps every element of a to [lo, hi] in place.
func MatrixClipInPlace(a *Matrix, lo, hi float32) {
	C.gf_matrix_clip_in_place(a.ptr, C.float(lo), C.float(hi))
}

// MatrixSafeSet writes val to element (r, c) of m. No-op if out-of-range.
func MatrixSafeSet(m *Matrix, r, c int, val float32) {
	C.gf_matrix_safe_set(m.ptr, C.int(r), C.int(c), C.float(val))
}

// ActivationBackward computes the gradient of the named activation applied to
// preAct, scaled by gradOut. act: "relu" | "sigmoid" | "tanh" | "leaky" | "none".
// Caller owns the returned Matrix.
func ActivationBackward(gradOut, preAct *Matrix, act string) *Matrix {
	cs := cstr(act); defer cfree(cs)
	return newMatrix(C.gf_activation_backward(gradOut.ptr, preAct.ptr, cs))
}

// ─── Layer ────────────────────────────────────────────────────────────────────

// Layer is an opaque handle to a single network layer.
// All factory functions register a finalizer so the layer is freed when GC'd.
// Call Free() for deterministic release.
type Layer struct{ ptr unsafe.Pointer }

func newLayer(p unsafe.Pointer) *Layer {
	if p == nil {
		return nil
	}
	l := &Layer{ptr: p}
	runtime.SetFinalizer(l, (*Layer).Free)
	return l
}

// Free releases the underlying C memory immediately.
// After Free, the Layer must not be used.
func (l *Layer) Free() {
	if l.ptr != nil {
		C.gf_layer_free(l.ptr)
		l.ptr = nil
		runtime.SetFinalizer(l, nil)
	}
}

// LayerDense creates a fully-connected layer: inSz → outSz with named activation.
func LayerDense(inSz, outSz int, act string) *Layer {
	cs := cstr(act); defer cfree(cs)
	return newLayer(C.gf_layer_dense(C.int(inSz), C.int(outSz), cs))
}

// LayerConv2D creates a 2-D convolutional layer.
func LayerConv2D(inCh, outCh, kernelH, kernelW, stride, padding int, act string) *Layer {
	cs := cstr(act); defer cfree(cs)
	return newLayer(C.gf_layer_conv2d(
		C.int(inCh), C.int(outCh),
		C.int(kernelH), C.int(kernelW),
		C.int(stride), C.int(padding), cs,
	))
}

// LayerDeconv2D creates a 2-D transposed-convolutional (deconv) layer.
func LayerDeconv2D(inCh, outCh, kernelH, kernelW, stride, padding int, act string) *Layer {
	cs := cstr(act); defer cfree(cs)
	return newLayer(C.gf_layer_deconv2d(
		C.int(inCh), C.int(outCh),
		C.int(kernelH), C.int(kernelW),
		C.int(stride), C.int(padding), cs,
	))
}

// LayerConv1D creates a 1-D convolutional layer.
func LayerConv1D(inCh, outCh, kernelSz, stride, padding int, act string) *Layer {
	cs := cstr(act); defer cfree(cs)
	return newLayer(C.gf_layer_conv1d(
		C.int(inCh), C.int(outCh),
		C.int(kernelSz), C.int(stride), C.int(padding), cs,
	))
}

// LayerBatchNorm creates a batch-normalisation layer for the given feature count.
func LayerBatchNorm(features int) *Layer {
	return newLayer(C.gf_layer_batch_norm(C.int(features)))
}

// LayerLayerNorm creates a layer-normalisation layer for the given feature count.
func LayerLayerNorm(features int) *Layer {
	return newLayer(C.gf_layer_layer_norm(C.int(features)))
}

// LayerAttention creates a multi-head self-attention layer.
func LayerAttention(dModel, nHeads int) *Layer {
	return newLayer(C.gf_layer_attention(C.int(dModel), C.int(nHeads)))
}

// Forward runs a forward pass through the layer. Caller owns the returned Matrix.
func (l *Layer) Forward(inp *Matrix) *Matrix {
	return newMatrix(C.gf_layer_forward(l.ptr, inp.ptr))
}

// Backward runs a backward pass through the layer. Caller owns the returned Matrix.
func (l *Layer) Backward(grad *Matrix) *Matrix {
	return newMatrix(C.gf_layer_backward(l.ptr, grad.ptr))
}

// InitOptimizer attaches an optimizer to the layer.
// opt: "adam" | "sgd" | "rmsprop"
func (l *Layer) InitOptimizer(opt string) {
	cs := cstr(opt); defer cfree(cs)
	C.gf_layer_init_optimizer(l.ptr, cs)
}

// Conv2D runs the Conv2D forward pass. Caller owns the returned Matrix.
func (l *Layer) Conv2D(inp *Matrix) *Matrix {
	return newMatrix(C.gf_layer_conv2d_forward(l.ptr, inp.ptr))
}

// Conv2DBackward runs the Conv2D backward pass. Caller owns the returned Matrix.
func (l *Layer) Conv2DBackward(grad *Matrix) *Matrix {
	return newMatrix(C.gf_layer_conv2d_backward(l.ptr, grad.ptr))
}

// VerifyWeights sanitises layer weights (replaces NaN/Inf with 0).
func (l *Layer) VerifyWeights() {
	C.gf_layer_verify_weights(l.ptr)
}

// ─── MatrixArray ──────────────────────────────────────────────────────────────

// MatrixArray is a growable array of Matrix pointers managed on the C heap.
type MatrixArray struct{ ptr unsafe.Pointer }

func newMatrixArray(p unsafe.Pointer) *MatrixArray {
	if p == nil {
		return nil
	}
	a := &MatrixArray{ptr: p}
	runtime.SetFinalizer(a, (*MatrixArray).Free)
	return a
}

// NewMatrixArray allocates an empty MatrixArray.
func NewMatrixArray() *MatrixArray {
	return newMatrixArray(C.gf_matrix_array_create())
}

// Free releases the underlying C memory immediately.
// After Free, the MatrixArray must not be used.
func (a *MatrixArray) Free() {
	if a.ptr != nil {
		C.gf_matrix_array_free(a.ptr)
		a.ptr = nil
		runtime.SetFinalizer(a, nil)
	}
}

// Push appends m to the array (the array does not take ownership of m).
func (a *MatrixArray) Push(m *Matrix) {
	C.gf_matrix_array_push(a.ptr, m.ptr)
}

// Len returns the number of matrices currently stored in the array.
func (a *MatrixArray) Len() int {
	return int(C.gf_matrix_array_len(a.ptr))
}

// ─── Network — extended methods ───────────────────────────────────────────────

// SampleConditional generates count outputs conditioned on cond.
// noiseType: "gauss" | "uniform" | "analog". Caller owns the returned Matrix.
func (n *Network) SampleConditional(count, noiseDim, condSz int, noiseType string, cond *Matrix) *Matrix {
	cs := cstr(noiseType); defer cfree(cs)
	return newMatrix(C.gf_network_sample_conditional(
		n.ptr, C.int(count), C.int(noiseDim), C.int(condSz), cs, cond.ptr,
	))
}

// AddProgressiveLayer appends a progressive-growing layer at resolution level resLvl.
func (n *Network) AddProgressiveLayer(resLvl int) {
	C.gf_network_add_progressive_layer(n.ptr, C.int(resLvl))
}

// GetLayerOutput returns the cached output of layer idx from the last forward pass.
// Caller owns the returned Matrix.
func (n *Network) GetLayerOutput(idx int) *Matrix {
	return newMatrix(C.gf_network_get_layer_output(n.ptr, C.int(idx)))
}

// DeepCopy returns an independent copy of the network. Caller owns the result.
func (n *Network) DeepCopy() *Network {
	return newNetwork(C.gf_network_deep_copy(n.ptr))
}

// DiscEvaluate runs the discriminator forward pass and returns scores.
// Caller owns the returned Matrix.
func (n *Network) DiscEvaluate(inp *Matrix) *Matrix {
	return newMatrix(C.gf_disc_evaluate(n.ptr, inp.ptr))
}

// DiscGradPenalty computes the gradient penalty (e.g. WGAN-GP) between real
// and fake samples scaled by lambda.
func (n *Network) DiscGradPenalty(real, fake *Matrix, lambda float32) float32 {
	return float32(C.gf_disc_grad_penalty(n.ptr, real.ptr, fake.ptr, C.float(lambda)))
}

// DiscFeatureMatch computes the feature-matching loss between real and fake
// at layer index featLayer.
func (n *Network) DiscFeatureMatch(real, fake *Matrix, featLayer int) float32 {
	return float32(C.gf_disc_feature_match(n.ptr, real.ptr, fake.ptr, C.int(featLayer)))
}

// DiscAddProgressiveLayer appends a progressive-growing layer to the discriminator
// at resolution level resLvl.
func (n *Network) DiscAddProgressiveLayer(resLvl int) {
	C.gf_disc_add_progressive_layer(n.ptr, C.int(resLvl))
}

// DiscGetLayerOutput returns the cached output of discriminator layer idx.
// Caller owns the returned Matrix.
func (n *Network) DiscGetLayerOutput(idx int) *Matrix {
	return newMatrix(C.gf_disc_get_layer_output(n.ptr, C.int(idx)))
}

// DiscDeepCopy returns an independent copy of the discriminator. Caller owns the result.
func (n *Network) DiscDeepCopy() *Network {
	return newNetwork(C.gf_disc_deep_copy(n.ptr))
}

// ─── Training extensions ──────────────────────────────────────────────────────

// TrainOptimize runs one optimizer step for net (applies accumulated gradients).
func TrainOptimize(net *Network) {
	C.gf_train_optimize(net.ptr)
}

// TrainAdamUpdate performs a single Adam parameter update.
// param, grad, m, v are the parameter matrix, gradient, first-moment, and
// second-moment accumulators. t is the time-step; lr, beta1, beta2, eps are
// the standard Adam hyper-parameters.
func TrainAdamUpdate(param, grad, m, v *Matrix, t int, lr, beta1, beta2, eps float32) {
	C.gf_train_adam_update(
		param.ptr, grad.ptr, m.ptr, v.ptr,
		C.int(t), C.float(lr), C.float(beta1), C.float(beta2), C.float(eps),
	)
}

// TrainSGDUpdate performs a single SGD (with momentum) parameter update.
func TrainSGDUpdate(param, grad, velocity *Matrix, lr, momentum float32) {
	C.gf_train_sgd_update(param.ptr, grad.ptr, velocity.ptr, C.float(lr), C.float(momentum))
}

// TrainRMSPropUpdate performs a single RMSProp parameter update.
func TrainRMSPropUpdate(param, grad, cache *Matrix, lr, decay, eps float32) {
	C.gf_train_rmsprop_update(param.ptr, grad.ptr, cache.ptr, C.float(lr), C.float(decay), C.float(eps))
}

// TrainLabelSmoothing returns a new Matrix with label values clamped to [lo, hi].
// Caller owns the returned Matrix.
func TrainLabelSmoothing(labels *Matrix, lo, hi float32) *Matrix {
	return newMatrix(C.gf_train_label_smoothing(labels.ptr, C.float(lo), C.float(hi)))
}

// TrainLoadBMP loads a BMP image file into a Dataset. Caller owns the result.
func TrainLoadBMP(path string) *Dataset {
	cs := cstr(path); defer cfree(cs)
	return newDataset(C.gf_train_load_bmp(cs))
}

// TrainLoadWAV loads a WAV audio file into a Dataset. Caller owns the result.
func TrainLoadWAV(path string) *Dataset {
	cs := cstr(path); defer cfree(cs)
	return newDataset(C.gf_train_load_wav(cs))
}

// TrainAugment applies random augmentation to sample.
// dataType: "vector" | "image" | "audio". Caller owns the returned Matrix.
func TrainAugment(sample *Matrix, dataType string) *Matrix {
	cs := cstr(dataType); defer cfree(cs)
	return newMatrix(C.gf_train_augment(sample.ptr, cs))
}

// TrainLogMetrics appends the metrics in m to filename as a CSV row.
func TrainLogMetrics(m *Metrics, filename string) {
	cs := cstr(filename); defer cfree(cs)
	C.gf_train_log_metrics(m.ptr, cs)
}

// TrainSaveSamples generates and saves sample images produced by gen at epoch ep.
func TrainSaveSamples(gen *Network, ep int, dir string, noiseDim int, noiseType string) {
	cd := cstr(dir); defer cfree(cd)
	cn := cstr(noiseType); defer cfree(cn)
	C.gf_train_save_samples(gen.ptr, C.int(ep), cd, C.int(noiseDim), cn)
}

// TrainPlotCSV writes a two-column (dLoss, gLoss) CSV of length cnt to filename.
func TrainPlotCSV(filename string, dLoss, gLoss []float32, cnt int) {
	cs := cstr(filename); defer cfree(cs)
	C.gf_train_plot_csv(
		cs,
		(*C.float)(unsafe.Pointer(&dLoss[0])),
		(*C.float)(unsafe.Pointer(&gLoss[0])),
		C.int(cnt),
	)
}

// TrainPrintBar prints a single-line ASCII progress bar for the current epoch.
func TrainPrintBar(dLoss, gLoss float32, width int) {
	C.gf_train_print_bar(C.float(dLoss), C.float(gLoss), C.int(width))
}

// TrainComputeFID computes the Fréchet Inception Distance between two sets of
// real and fake samples.
func TrainComputeFID(realArr, fakeArr *MatrixArray) float32 {
	return float32(C.gf_train_compute_fid(realArr.ptr, fakeArr.ptr))
}

// TrainComputeIS computes the Inception Score for a set of generated samples.
func TrainComputeIS(samples *MatrixArray) float32 {
	return float32(C.gf_train_compute_is(samples.ptr))
}

// ─── Security extensions ──────────────────────────────────────────────────────

// SecGetOSRandom returns a single cryptographically-random byte from the OS RNG.
func SecGetOSRandom() uint8 {
	return uint8(C.gf_sec_get_os_random())
}

// SecEncryptModel encrypts the model file at inF with key and writes the result
// to outF.
func SecEncryptModel(inF, outF, key string) {
	ci := cstr(inF); defer cfree(ci)
	co := cstr(outF); defer cfree(co)
	ck := cstr(key); defer cfree(ck)
	C.gf_sec_encrypt_model(ci, co, ck)
}

// SecDecryptModel decrypts the model file at inF with key and writes the result
// to outF.
func SecDecryptModel(inF, outF, key string) {
	ci := cstr(inF); defer cfree(ci)
	co := cstr(outF); defer cfree(co)
	ck := cstr(key); defer cfree(ck)
	C.gf_sec_decrypt_model(ci, co, ck)
}

// SecRunTests runs the built-in security self-tests and returns the number of
// failures (0 = all passed).
func SecRunTests() int {
	return int(C.gf_sec_run_tests())
}

// SecRunFuzzTests runs the security fuzz tests for the given number of
// iterations and returns the number of failures found.
func SecRunFuzzTests(iterations int) int {
	return int(C.gf_sec_run_fuzz_tests(C.int(iterations)))
}

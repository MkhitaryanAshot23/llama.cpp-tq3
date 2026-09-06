# Native SYCL TQ3_4S: correctness milestone

This implementation adds dense `GGML_OP_MUL_MAT` for `GGML_TYPE_TQ3_4S`
weights. It prioritizes single-token decode on Intel Arc B580/Xe2. The code and
model-free tests are prepared, but **C++ compilation and GPU execution have not
been validated in the development environment** (see the build record below).

## Format and reference choice

The existing type is `GGML_TYPE_TQ3_4S = 46`, distinct from
`GGML_TYPE_TQ3_0 = 200`. The existing `block_tq3_4s` is unchanged: four E3M5
scale bytes followed by twelve index bytes, representing 32 weights in 16 bytes.
No GGUF or quantization-type changes are made.

The numerical contract is the current `quantize_row_tq3_4s_ref` and
`dequantize_row_tq3_4s` in `ggml/src/ggml-quants.c`. These functions explicitly
use the `TQ3_0_CENTROIDS` and `TQ3_0_SIGNS` tables. Their use here follows those
TQ3_4S functions, rather than assuming TQ3_0 and TQ3_4S are interchangeable.

For group `g = 0..3`, three bytes at `qs + 3*g` form a little-endian 24-bit
integer. Index `j = 0..7` is `(packed >> (3*j)) & 7`, corresponding to weight
`8*g + j`. Exactly three bytes are read, including the last group of a block.
Each group's independent scale is

```text
s(d) = 0                                      if d == 0
       (1 + (d & 31)/32) * 2^((d >> 5) - 9)  otherwise
```

The helper computes the algebraically equivalent
`(32 + (d & 31)) * (1 << (d >> 5)) / 16384`, exactly representable in FP32.

## CUDA comparison, including normalization

The current CUDA `vec_dot_tq3_4s_q8_1` does not implement the CPU codebook
exactly, even after accounting for all its scale factors:

| Index | CPU encoder/decoder | CUDA int8 level | CUDA level times `2.1519/127` |
| --- | ---: | ---: | ---: |
| 0 | -1.996684 | -127 | -2.151900000 |
| 1 | -1.291398 | -79 | -1.338583465 |
| 2 | -0.740341 | -45 | -0.762484252 |
| 3 | -0.247508 | -14 | -0.237217323 |
| 4 | 0.230106 | 14 | 0.237217323 |
| 5 | 0.725222 | 45 | 0.762484252 |
| 6 | 1.277503 | 79 | 1.338583465 |
| 7 | 1.988943 | 127 | 2.151900000 |

CUDA's `tq3_4s_ratio4s` decodes the same E3M5 scale for every one of the 256
byte values. Both paths multiply by the Q8 activation scale. Both use the same
orthonormal RHT. CUDA's two dot calls cover two groups each; SYCL's four calls
cover one group each. Neither reduction introduces another normalization.

There is no common scalar that makes the codebooks equivalent: CUDA's table
is antisymmetric, while the CPU table is not. For example, for scale byte 255,
index 7 and a rotated Q8 activation with one nonzero value `q=1, d8=1`, the
contributions are `0.9789328828` (CPU) and `1.0591382813` (CUDA). This is a
codebook difference, not ordinary FP32 rounding or a missing RHT factor.

The SYCL path deliberately uses the exact CPU codebook. It does not change
CUDA, the CPU encoder, the model, or the format to conceal this discrepancy.
Parity with current CUDA MMVQ and compatibility with an externally quantized
model must therefore not be inferred from CPU-reference correctness.

## Activation transform and dot

For each group of 32 activations:

```text
r = H32 * Dsign * activation / sqrt(32)
d8 = max(abs(r)) / 127
q8[i] = round(r[i] / d8)   (zero when the block is zero)
result += sum_g s[g] * sum_j centroid[index[g,j]] * q8[8*g+j] * fp16(d8)
```

The 32 signs exactly equal CUDA's unsigned 32-bit golden-ratio hash
`((i * 0x9E3779B9u) >> 31) ? -1 : +1`. The five butterfly stages use natural
Hadamard order; there is no additional permutation. The CPU inverse is
`Dsign * H32 / sqrt(32)`, so applying the forward transform to activations
preserves the dot product with decoded weights before Q8 rounding.

`quantize_tq3_4s_q8_1` fuses RHT and Q8_1 quantization. It uses the existing
launcher with `reqd_sub_group_size(WARP_SIZE)`. For Intel's current
`WARP_SIZE=16`, a lane holds two adjacent values: the first butterfly is local,
and the remaining four use `dpct::permute_sub_group_by_xor` with an explicit
logical subgroup size. The implementation also supports a 32-lane build.
Capability checks verify GPU, FP16 and the required subgroup size.

The Q8_1 output retains the existing padded-row allocation and layout. Its
scale and activation sum are stored in `ds`; the dot uses `ds[0]`. Padding is
not read by TQ3_4S MMVQ. Strided activation rows are copied through the existing
SYCL path and use the same RHT quantizer afterward. Source activations are
not modified. There is no rotated-activation scratch allocation or dequantized
weight-matrix allocation.

`vec_dot_tq3_4s_q8_1` unpacks eight weights per call on the GPU and accumulates
in FP32. The generic MMVQ uses `qk=32`, `qi=8`, `vdr=2`: four lanes per weight
block, groups 0 through 3 exactly once. On a 16-lane subgroup it advances by
four blocks per iteration, including tail blocks. No DP4A approximation is used.

## Dispatch and scope

`ggml_sycl_mul_mat` selects the TQ3_4S quantizer and MMVQ before the generic
DMMV, reorder, and oneMKL paths. Additional activation columns use one GEMV
launch per column, including batches larger than `MMVQ_MAX_BATCH_SIZE`.
This permits basic prefill and columns 2..3 without adding a GEMM kernel.

Supported shapes are contiguous dense 2D TQ3_4S weights, 2D FP32 activations
with contiguous elements within each row (row strides are allowed), FP32
results, and a single SYCL device. `K` must be a multiple of 32.

The single-column offload check bypasses the generic minimum-batch threshold.
Once this supported operation enters SYCL, it has no CPU, DMMV, or full-weight
dequantization alternative. With `GGML_SYCL_DEBUG=1`, a message is printed once
per process when the path is first selected:

```text
TQ3_4S -> SYCL MMVQ (CPU-reference RHT + Q8_1, native GPU unpack)
```

This is not a global ban on the scheduler using other backends. Explicit CPU
placement and unsupported operations remain outside this implementation.
`MUL_MAT_ID`, `GET_ROWS`, batched/noncontiguous weights, tensor splitting,
and non-FP32 activations are not advertised as supported TQ3_4S operations.
Full-model GPU placement has not been tested.

## Model-free validation

`tests/test-tq3-4s.cpp` builds two executables:

- `test-tq3-4s`: actual C++ format helpers versus `dequantize_row_tq3_4s`, all
  256 scales, all 8 indices at every position, all centroid/scale combinations,
  random payloads, encoder-generated blocks, zero/random activations, and dots.
  Four seeds and `K = 32, 64, 96, 256, 512, 544, 1024, 4096` are covered.
- `test-tq3-4s-sycl` (when `GGML_SYCL=ON`): the above tests plus 480 GPU matmul
  cases: four seeds, five K sizes, rows 1/3/17, columns 1/2/3/9, contiguous and
  strided activation rows. It checks capability rejection, single-column
  offload eligibility, zero/impulse activations, untouched source buffers and
  complete output writes. It selects a SYCL GPU and calls
  `ggml_backend_graph_compute` directly, without a scheduler. No GPU yields an
  explicit CTest skip (77), not a CPU pass; an unavailable explicitly named
  device is a failure.

Reference rotation uses an independently constructed dense Hadamard matrix.
For GPU comparison with the quantized reference, absolute tolerance is
`2e-5 * max(1, sum(abs(rotated_weight * reconstructed_Q8_activation)))`.
This allows FP32 accumulation order differences without dividing by a nearly
zero dot product. The CPU helper comparison uses `2e-6` instead.

Comparison with the original FP32 activation additionally permits the
per-block bound
`sum(abs(rotated_weight)) * (d8/2 + 127*abs(d8 - fp16(d8)))`.
This accounts separately for Q8 rounding and FP16 scale storage. It does not
use a broad tolerance to hide codebook or sign errors.

`tests/test-tq3-4s-math.py` provides standard-library-only numerical and source
checks without a compiler. It simulates both subgroup widths in FP32 and
compares against an explicit Hadamard matrix, including all basis vectors,
zero/random/small activations, padded rows and MMVQ tail-block coverage. It
also reads and audits the current CUDA packed codebook and scale normalization.
These checks do not execute compiled C++ or a SYCL kernel.

## Build and run after the external toolchain is available

From an existing configured Windows MSVC/oneAPI development environment:

```powershell
cmake -S . -B build-tq3-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DLLAMA_BUILD_TESTS=ON
cmake --build build-tq3-sycl --target test-tq3-4s test-tq3-4s-sycl llama-cli
$env:ONEAPI_DEVICE_SELECTOR = "level_zero:gpu"
$env:GGML_SYCL_DEBUG = "1"
ctest --test-dir build-tq3-sycl -R '^test-tq3-4s(-sycl)?$' --output-on-failure
```

Optionally pass an exact SYCL device name as the first argument to
`test-tq3-4s-sycl` to require that device. Confirm the printed device description
identifies the intended B580. The commands above run no model tests/downloads.

Without a compiler:

```powershell
python tests/test-tq3-4s-math.py
```

## Development environment record (2026-09-07)

- Working branch: `tq3-sycl-b580`, based on local `origin/main` at `47635d7`.
  The initial unborn branch was attached to this commit with the working
  files preserved; the resulting baseline was clean. `main` was not changed.
- Python 3.12.14: all seven numerical/source tests pass.
- `git diff --check`: passes.
- CMake configure could not start: `cmake` is not recognized. `where.exe`
  could not locate CMake, Ninja, icx, icpx, dpcpp, cl or clang++, including in
  the normal host environment. Standard CMake/Intel oneAPI/LLVM installation
  paths and the Visual Studio Installer locator were also absent.
- Consequently neither C++ test executable, the SYCL backend, nor a GPU test
  could be built or run. This is an external toolchain blocker, not a recorded
  compiler diagnostic. No software, SDK or driver was installed.
- Local logs are in the ignored `build-tq3-sycl-milestone/math-tests.log` and
  `build-tq3-sycl-milestone/configure.log`.
- No GGUF was searched for, downloaded, created, converted, or executed. The
  future Qwen3.8-27B-TQ3_4S-v2 model remains unvalidated.

The next step is a real toolchain build and these model-free GPU tests. Only
after correctness is confirmed should performance work consider reusing weight
loads across columns, reducing launch overhead, or a suitable GEMM path.
Any int8 optimization must first preserve the chosen codebook contract.

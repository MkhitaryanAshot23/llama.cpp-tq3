// Model-free tests for the TQ3_4S weight format and native SYCL MUL_MAT.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-quants.h"
#include "ggml-sycl/tq3_4s.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace tq = ggml_sycl_tq3_4s;

static void check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static void close(double actual, double expected, double tolerance, const char * message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::fprintf(stderr, "%s: actual=%.10g expected=%.10g tolerance=%.6g\n",
                     message, actual, expected, tolerance);
        check(false, message);
    }
}

// Independent reference: explicit Hadamard matrix, no subgroup/butterfly indexing.
static int hadamard(int i, int j) {
    unsigned bits = unsigned(i & j);
    int result = 1;
    while (bits) {
        result = -result;
        bits &= bits - 1;
    }
    return result;
}

static constexpr float ref_signs[32] = {
    +1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1,
    -1, -1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1,
};

static std::array<double, 32> rotate(const float * x) {
    std::array<double, 32> out{};
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            out[i] += hadamard(i, j) * ref_signs[j] * x[j] / std::sqrt(32.0);
        }
    }
    return out;
}

static void pack(block_tq3_4s & block, const std::array<uint8_t, 32> & indices) {
    std::memset(block.qs, 0, sizeof(block.qs));
    // Bit-at-a-time packing, independent of the production 24-bit extraction.
    for (int i = 0; i < 32; ++i) {
        for (int bit = 0; bit < 3; ++bit) {
            const int pos = 3 * i + bit;
            block.qs[pos / 8] |= ((indices[i] >> bit) & 1) << (pos % 8);
        }
    }
}

static void test_format() {
    check(GGML_TYPE_TQ3_4S != GGML_TYPE_TQ3_0, "distinct weight formats");
    check(ggml_blck_size(GGML_TYPE_TQ3_4S) == 32 && sizeof(block_tq3_4s) == 16, "block layout");
    for (int d = 0; d < 256; ++d) {
        const float expected = d == 0 ? 0.0f : std::ldexp(1.0f + (d & 31) / 32.0f, (d >> 5) - 9);
        close(tq::scale(d), expected, 0, "all E3M5 encodings");
    }
    for (int pos = 0; pos < 32; ++pos) {
        for (int value = 0; value < 8; ++value) {
            std::array<uint8_t, 32> indices{};
            indices[pos] = value;
            block_tq3_4s block{};
            pack(block, indices);
            for (int j = 0; j < 32; ++j) {
                check(tq::index(tq::unpack_group(block.qs, j / 8), j % 8) == indices[j], "index order/boundaries");
            }
        }
    }
    // Recover every centroid/scale combination from the actual CPU decoder.
    for (int d = 0; d < 256; ++d) {
        for (int idx = 0; idx < 8; ++idx) {
            block_tq3_4s block{};
            std::array<uint8_t, 32> indices{};
            indices.fill(uint8_t(idx));
            pack(block, indices);
            for (int g = 0; g < 4; ++g) {
                block.d[g] = uint8_t((d + 53 * g) % 256);
            }
            float logical[32];
            dequantize_row_tq3_4s(&block, logical, 32);
            const auto rotated = rotate(logical);
            for (int i = 0; i < 32; ++i) {
                close(rotated[i], tq::centroid(idx) * tq::scale(block.d[i / 8]), 3e-7, "CPU decoder/codebook/scales/signs");
                close(tq::sign(i), ref_signs[i], 0, "sign table");
            }
        }
    }
}

struct activation_block {
    std::array<int8_t, 32> qs;
    float d;
    float raw_d;
};

static activation_block quantize_activation(const float * x) {
    // FP32 CPU butterfly order, so Q8 rounding boundaries match the device.
    float v[32];
    for (int i = 0; i < 32; ++i) {
        v[i] = x[i] * ref_signs[i];
    }
    for (int step = 1; step < 32; step *= 2) {
        for (int base = 0; base < 32; base += 2 * step) {
            for (int j = 0; j < step; ++j) {
                const float a = v[base + j], b = v[base + j + step];
                v[base + j] = a + b;
                v[base + j + step] = a - b;
            }
        }
    }
    const auto matrix = rotate(x);
    float amax = 0;
    for (int i = 0; i < 32; ++i) {
        v[i] *= 1.0f / std::sqrt(32.0f);
        close(v[i], matrix[i], 2e-6 * std::max(1.0, std::abs(matrix[i])), "RHT normalization/order");
        amax = std::max(amax, std::abs(v[i]));
    }
    activation_block out{};
    out.raw_d = amax / 127.0f;
    out.d = ggml_fp16_to_fp32(ggml_fp32_to_fp16(out.raw_d));
    for (int i = 0; i < 32; ++i) {
        out.qs[i] = amax == 0 ? 0 : int8_t(std::round(v[i] / out.raw_d));
    }
    return out;
}

struct dot_reference {
    double quantized = 0;
    double original = 0;
    double q8_error_bound = 0;
    double magnitude = 0;
};

static dot_reference reference_dot(const block_tq3_4s * weights, const float * activation, int k) {
    dot_reference result;
    for (int b = 0; b < k / 32; ++b) {
        float w[32];
        dequantize_row_tq3_4s(weights + b, w, 32);
        const auto rotated = rotate(w);
        const auto q = quantize_activation(activation + 32 * b);
        double helper_dot = 0;
        for (int g = 0; g < 4; ++g) {
            helper_dot += tq::dot_group(weights[b].d, weights[b].qs, q.qs.data(), g) * q.d;
        }
        double block_dot = 0, magnitude = 0;
        for (int i = 0; i < 32; ++i) {
            const double product = rotated[i] * q.qs[i] * q.d;
            block_dot += product;
            magnitude += std::abs(product);
            result.original += double(w[i]) * activation[32 * b + i];
            // Half-step Q8 rounding plus storage of d as FP16.
            result.q8_error_bound += std::abs(rotated[i]) * (q.raw_d / 2.0 + 127 * std::abs(q.raw_d - q.d));
        }
        close(helper_dot, block_dot, 2e-6 * std::max(1.0, magnitude), "native dot math vs CPU decoder");
        result.quantized += block_dot;
        result.magnitude += magnitude;
    }
    close(result.quantized, result.original,
          result.q8_error_bound + 2e-6 * std::max(1.0, result.magnitude), "Q8 analytical error bound");
    return result;
}

static void random_weights(std::mt19937 & rng, std::vector<block_tq3_4s> & w) {
    for (auto & block : w) {
        for (auto & d : block.d) { d = uint8_t(rng() % 256); }
        for (auto & q : block.qs) { q = uint8_t(rng() % 256); }
    }
}

#ifdef TEST_TQ3_4S_SYCL
static void test_capabilities(ggml_backend_t backend) {
    auto * ctx = ggml_init({ggml_tensor_overhead() * 32, nullptr, true});
    check(ctx != nullptr, "capability test context");
    auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ3_4S, 32, 3);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 1);
    check(ggml_backend_supports_op(backend, ggml_mul_mat(ctx, w, x)), "dense decode supported");
    auto * batched_w = ggml_new_tensor_3d(ctx, GGML_TYPE_TQ3_4S, 32, 3, 2);
    auto * batched_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 1, 2);
    check(!ggml_backend_supports_op(backend, ggml_mul_mat(ctx, batched_w, batched_x)), "batched weights not advertised");
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    check(!ggml_backend_supports_op(backend, ggml_get_rows(ctx, w, ids)), "GET_ROWS not advertised");
    auto * strided_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ3_4S, 64, 3);
    auto * strided_w = ggml_view_2d(ctx, strided_storage, 32, 3, strided_storage->nb[1], 0);
    check(!ggml_backend_supports_op(backend, ggml_mul_mat(ctx, strided_w, x)), "strided weights not advertised");
    ggml_free(ctx);
}

static void test_matmul(ggml_backend_t backend, std::mt19937 & rng, int k, int rows, int cols, bool strided) {
    ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true});
    check(ctx != nullptr, "context allocation");
    auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ3_4S, k, rows);
    const int pitch = k + (strided ? 32 : 0);
    const int offset = strided ? 16 : 0;
    auto * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pitch, cols);
    auto * x = strided ? ggml_view_2d(ctx, storage, k, cols, storage->nb[1], offset * sizeof(float)) : storage;
    auto * y = ggml_mul_mat(ctx, w, x);
    ggml_set_name(w, "tq3_4s_test_weights");
    ggml_set_name(y, "tq3_4s_test_matmul");
    check(ggml_backend_supports_op(backend, y), "SYCL must claim dense TQ3_4S MUL_MAT");
    check(ggml_backend_dev_offload_op(ggml_backend_get_device(backend), y), "single-column offload eligibility");
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "GPU buffer allocation");
    std::vector<block_tq3_4s> weights(size_t(rows) * k / 32);
    random_weights(rng, weights);
    std::vector<float> input(size_t(pitch) * cols, -12345.0f);
    std::normal_distribution<float> normal;
    for (int c = 0; c < cols; ++c) {
        for (int i = 0; i < k; ++i) {
            input[size_t(c) * pitch + offset + i] = normal(rng);
        }
    }
    // Also exercise zero and impulse activations, including Q8 d == 0.
    if (cols > 1) {
        std::fill_n(input.data() + pitch + offset, k, 0.0f);
    }
    if (cols > 2) {
        std::fill_n(input.data() + 2 * pitch + offset, k, 0.0f);
        input[2 * pitch + offset + 23] = -1.0f;
    }
    std::vector<float> output(size_t(rows) * cols, std::numeric_limits<float>::quiet_NaN());
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * sizeof(block_tq3_4s));
    ggml_backend_tensor_set(storage, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(y, output.data(), 0, output.size() * sizeof(float));
    // Direct execution on the selected SYCL backend: no scheduler or CPU backend.
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "SYCL graph compute");
    ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
    for (int c = 0; c < cols; ++c) {
        for (int row = 0; row < rows; ++row) {
            const auto ref = reference_dot(weights.data() + size_t(row) * k / 32,
                                           input.data() + size_t(c) * pitch + offset, k);
            const double tol = 2e-5 * std::max(1.0, ref.magnitude);
            close(output[size_t(c) * rows + row], ref.quantized, tol, "GPU vs quantized CPU reference");
            close(output[size_t(c) * rows + row], ref.original, ref.q8_error_bound + tol, "GPU vs FP32 activation");
        }
    }
    std::vector<float> after(input.size());
    ggml_backend_tensor_get(storage, after.data(), 0, after.size() * sizeof(float));
    check(after == input, "RHT must not modify source activations");
    std::printf("PASS SYCL k=%d rows=%d cols=%d strided=%d\n", k, rows, cols, int(strided));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}
#endif

int main(int argc, char ** argv) {
    auto * init = ggml_init({ggml_tensor_overhead(), nullptr, true});
    check(init != nullptr, "ggml initialization");
    ggml_free(init);
    test_format();
    for (unsigned seed : {1u, 42u, 2026u, 0xdeadbeefu}) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal;
        for (int k : {32, 64, 96, 256, 512, 544, 1024, 4096}) {
            std::vector<block_tq3_4s> w(k / 32);
            random_weights(rng, w);
            std::vector<float> x(k);
            for (auto & v : x) { v = normal(rng); }
            reference_dot(w.data(), x.data(), k);
            std::fill(x.begin(), x.end(), 0.0f);
            reference_dot(w.data(), x.data(), k);
            // Encoder-generated blocks as well as arbitrary payloads.
            for (auto & v : x) { v = normal(rng) * 0.1f; }
            quantize_row_tq3_4s_ref(x.data(), w.data(), k);
            reference_dot(w.data(), x.data(), k);
        }
    }
    std::puts("PASS TQ3_4S format, CPU decoder and dot math (4 seeds, 8 dimensions)");
#ifdef TEST_TQ3_4S_SYCL
    ggml_backend_load_all();
    auto reg = ggml_backend_reg_by_name("SYCL");
    ggml_backend_dev_t device = nullptr;
    if (reg) {
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
            auto candidate = ggml_backend_reg_dev_get(reg, i);
            if (ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                (argc < 2 || std::strcmp(argv[1], ggml_backend_dev_name(candidate)) == 0)) {
                device = candidate;
                break;
            }
        }
    }
    if (!device) {
        std::fprintf(stderr, "SKIP: no requested SYCL GPU; no CPU fallback permitted\n");
        return argc < 2 ? 77 : 1;
    }
    auto backend = ggml_backend_dev_init(device, nullptr);
    check(backend != nullptr, "SYCL backend initialization");
    std::printf("Testing %s: %s\n", ggml_backend_dev_name(device), ggml_backend_dev_description(device));
    test_capabilities(backend);
    for (unsigned seed : {1u, 42u, 2026u, 0xdeadbeefu}) {
        std::mt19937 rng(seed);
        for (int k : {32, 96, 256, 544, 4096}) {
            for (int rows : {1, 3, 17}) {
                for (int cols : {1, 2, 3, 9}) {
                    test_matmul(backend, rng, k, rows, cols, false);
                    test_matmul(backend, rng, k, rows, cols, true);
                }
            }
        }
    }
    ggml_backend_free(backend);
#else
    (void) argc;
    (void) argv;
#endif
    return 0;
}

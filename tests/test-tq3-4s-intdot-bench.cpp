// Isolated GPU-resident TQ3_4S SYCL microbenchmark.
// Measures repeated single-column MUL_MAT after weights/input are resident on the GPU.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-quants.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static ggml_backend_dev_t find_sycl_gpu(const char * requested) {
    ggml_backend_load_all();
    auto reg = ggml_backend_reg_by_name("SYCL");
    if (!reg) {
        return nullptr;
    }
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
        auto dev = ggml_backend_reg_dev_get(reg, i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        if (!requested || std::strcmp(requested, ggml_backend_dev_name(dev)) == 0) {
            return dev;
        }
    }
    return nullptr;
}

static void bench_case(ggml_backend_t backend, int k, int rows, int warmup, int iterations) {
    check(k % 32 == 0, "k must be divisible by TQ3_4S block size");

    auto * ctx = ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true});
    check(ctx != nullptr, "context allocation");

    auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ3_4S, k, rows);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    auto * y = ggml_mul_mat(ctx, w, x);
    ggml_set_name(w, "tq3_4s_bench_weights");
    ggml_set_name(x, "tq3_4s_bench_input");
    ggml_set_name(y, "tq3_4s_bench_output");

    check(ggml_backend_supports_op(backend, y), "SYCL must support TQ3_4S MUL_MAT");

    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "GPU buffer allocation");

    const size_t nblocks = size_t(rows) * size_t(k) / 32;
    std::vector<block_tq3_4s> weights(nblocks);
    std::vector<float> input(static_cast<size_t>(k));
    std::mt19937 rng(0x54493334u);
    std::normal_distribution<float> normal;
    for (auto & block : weights) {
        for (auto & d : block.d) {
            d = uint8_t(rng() % 256);
        }
        for (auto & q : block.qs) {
            q = uint8_t(rng() % 256);
        }
    }
    for (auto & value : input) {
        value = normal(rng);
    }

    // Host -> GPU transfers happen once, before warmup and before timing.
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * sizeof(block_tq3_4s));
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_synchronize(backend);

    for (int i = 0; i < warmup; ++i) {
        check(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS, "warmup graph compute");
    }
    ggml_backend_synchronize(backend);

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        check(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS, "timed graph compute");
    }
    ggml_backend_synchronize(backend);
    const auto stop = std::chrono::steady_clock::now();

    float sample = 0.0f;
    ggml_backend_tensor_get(y, &sample, 0, sizeof(sample));
    check(std::isfinite(sample), "benchmark output must be finite");

    const double total_s = std::chrono::duration<double>(stop - start).count();
    const double ms = total_s * 1000.0 / iterations;
    const double weight_bytes = double(weights.size() * sizeof(block_tq3_4s));
    const double gbps = (weight_bytes * iterations / total_s) / 1.0e9;

    std::printf("TQ3_4S GPU-resident: k=%d rows=%d weights=%.3f MB iter=%d time=%.4f ms effective=%.2f GB/s sample=%g\n",
                k, rows, weight_bytes / 1.0e6, iterations, ms, gbps, sample);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const char * requested = argc > 1 ? argv[1] : nullptr;
    auto device = find_sycl_gpu(requested);
    if (!device) {
        std::fprintf(stderr, "FAIL: no requested SYCL GPU; no CPU fallback permitted\n");
        return 1;
    }

    auto backend = ggml_backend_dev_init(device, nullptr);
    check(backend != nullptr, "SYCL backend initialization");
    std::printf("Benchmarking %s: %s\n", ggml_backend_dev_name(device), ggml_backend_dev_description(device));
    std::puts("Timed interval contains GPU graph execution only; H2D/D2H transfers are outside timing.");

    bench_case(backend, 5120, 17408, 20, 200);

    ggml_backend_free(backend);
    return 0;
}

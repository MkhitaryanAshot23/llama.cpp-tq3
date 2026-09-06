// TQ3_4S weight-format math, also usable by the host correctness tests.
#pragma once

#include <cstdint>

namespace ggml_sycl_tq3_4s {

constexpr int qk = 32;
// Four lanes per block, each processing one group of eight weights.
constexpr int qi = qk / 4;
constexpr int vdr = 2;
constexpr float rht_norm = 0.17677669529663687f;

inline float sign(const int i) {
    // The table used by quantize_row_tq3_4s_ref/dequantize_row_tq3_4s.
    // Also equals CUDA's uint32_t golden-ratio hash for all 32 indices.
    constexpr float signs[qk] = {
        +1, -1, +1, -1, +1, +1, -1, +1,
        -1, -1, +1, -1, +1, +1, -1, +1,
        -1, -1, +1, -1, +1, -1, -1, +1,
        -1, +1, +1, -1, +1, -1, -1, +1,
    };
    return signs[i];
}

inline float centroid(const uint8_t index) {
    // Exact CPU encoder/decoder codebook. CUDA MMVQ's int8 approximation
    // uses a different codebook and is not the reference for this path.
    constexpr float centroids[8] = {
        -1.996684f, -1.291398f, -0.740341f, -0.247508f,
         0.230106f,  0.725222f,  1.277503f,  1.988943f,
    };
    return centroids[index];
}

inline float scale(const uint8_t d) {
    // E3M5, exponent bias 9; byte zero is a special zero, not 2^-9.
    // This expression is exact in FP32 for all 256 encodings.
    return d == 0 ? 0.0f : float((32 + (d & 31)) * (1u << (d >> 5))) / 16384.0f;
}

inline uint32_t unpack_group(const uint8_t * qs, const int group) {
    const uint8_t * p = qs + 3 * group;
    // Read exactly three bytes, including for the last group of the last block.
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16);
}

inline uint8_t index(const uint32_t packed, const int j) {
    return (packed >> (3 * j)) & 7;
}

inline float dot_group(const uint8_t * d, const uint8_t * qs, const int8_t * activation, const int group) {
    const uint32_t packed = unpack_group(qs, group);
    float sum = 0.0f;
    for (int j = 0; j < 8; ++j) {
        sum += centroid(index(packed, j)) * float(activation[8 * group + j]);
    }
    return scale(d[group]) * sum;
}

} // namespace ggml_sycl_tq3_4s

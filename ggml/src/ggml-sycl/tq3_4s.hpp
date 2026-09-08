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

inline float centroid_decode(const uint8_t index) {
    // Select scalar constants instead of indexing a private array in decode.
    // Tiled multi-column MMVQ retains centroid() and reuses decoded weights.
    const bool b0 = (index & 1u) != 0;
    const bool b1 = (index & 2u) != 0;
    const bool b2 = (index & 4u) != 0;
    const float c01 = b0 ? -1.291398f : -1.996684f;
    const float c23 = b0 ? -0.247508f : -0.740341f;
    const float c45 = b0 ?  0.725222f :  0.230106f;
    const float c67 = b0 ?  1.988943f :  1.277503f;
    return b2 ? (b1 ? c67 : c45) : (b1 ? c23 : c01);
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
    const int8_t * a = activation + 8 * group;
    const float s0 = centroid_decode(index(packed, 0)) * float(a[0]) + centroid_decode(index(packed, 1)) * float(a[1]);
    const float s1 = centroid_decode(index(packed, 2)) * float(a[2]) + centroid_decode(index(packed, 3)) * float(a[3]);
    const float s2 = centroid_decode(index(packed, 4)) * float(a[4]) + centroid_decode(index(packed, 5)) * float(a[5]);
    const float s3 = centroid_decode(index(packed, 6)) * float(a[6]) + centroid_decode(index(packed, 7)) * float(a[7]);
    const float sum = (s0 + s1) + (s2 + s3);
    return scale(d[group]) * sum;
}

} // namespace ggml_sycl_tq3_4s

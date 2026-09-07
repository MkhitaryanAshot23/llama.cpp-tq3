// TQ3_4S weight-format math, also usable by the host correctness tests.
#pragma once

#include <cstdint>

#if defined(__SYCL_DEVICE_ONLY__)
#include "dpct/helper.hpp"
#endif

namespace ggml_sycl_tq3_4s {

constexpr int qk = 32;
// Four lanes per block, each processing one group of eight weights.
constexpr int qi = qk / 4;
constexpr int vdr = 2;
constexpr float rht_norm = 0.17677669529663687f;

// Exact CPU encoder/decoder codebook. Keep this named table as the format
// source-of-truth for host/source correctness tests.
constexpr float centroids[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f,
};

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
    // Express the exact lookup as a balanced select tree so LLVM can lower it
    // to predicated selects on Xe rather than an indirect table load or a
    // divergent 8-way switch. CUDA's int8 approximation is deliberately not
    // used here.
    const bool b0 = (index & 1u) != 0;
    const bool b1 = (index & 2u) != 0;
    const bool b2 = (index & 4u) != 0;

    const float c01 = b0 ? centroids[1] : centroids[0];
    const float c23 = b0 ? centroids[3] : centroids[2];
    const float c45 = b0 ? centroids[5] : centroids[4];
    const float c67 = b0 ? centroids[7] : centroids[6];
    const float c03 = b1 ? c23 : c01;
    const float c47 = b1 ? c67 : c45;
    return b2 ? c47 : c03;
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

#if defined(__SYCL_DEVICE_ONLY__)

// Each exact six-decimal centroid is represented as an integer multiplied by
// 1e6, then decomposed into three signed base-256 digits. This lets Xe2 use
// three integer dot-product planes instead of eight FP32 centroid multiplies.
// These are NOT CUDA's approximate {-127,-79,-45,-14,14,45,79,127} levels.
// Low/mid/high bytes are signed base-256 digits respectively.
inline uint32_t centroid_digits(const uint8_t idx) {
    const bool b0 = (idx & 1u) != 0;
    const bool b1 = (idx & 2u) != 0;
    const bool b2 = (idx & 4u) != 0;

    // Exact integer centroids and their balanced signed-byte decomposition:
    // -1996684 -> { 116,-120,-30 }, -1291398 -> { 122,  75,-20 }
    //  -740341 -> {  11, -76,-11 },  -247508 -> {  44,  57, -4 }
    //   230106 -> { -38,-125,  4 },   725222 -> { -26,  17, 11 }
    //  1277503 -> {  63, 126, 19 },  1988943 -> {  79,  89, 30 }
    const uint32_t c01 = b0 ? 0x00EC4B7Au : 0x00E28874u;
    const uint32_t c23 = b0 ? 0x00FC392Cu : 0x00F5B40Bu;
    const uint32_t c45 = b0 ? 0x000B11E6u : 0x000483DAu;
    const uint32_t c67 = b0 ? 0x001E594Fu : 0x00137E3Fu;
    const uint32_t c03 = b1 ? c23 : c01;
    const uint32_t c47 = b1 ? c67 : c45;
    return b2 ? c47 : c03;
}

inline uint32_t pack_i8x4(const int8_t * p) {
    return uint32_t(uint8_t(p[0]))
         | (uint32_t(uint8_t(p[1])) << 8)
         | (uint32_t(uint8_t(p[2])) << 16)
         | (uint32_t(uint8_t(p[3])) << 24);
}

inline uint32_t pack_digit_plane(const uint32_t c0, const uint32_t c1,
                                 const uint32_t c2, const uint32_t c3,
                                 const int byte_shift) {
    return ((c0 >> byte_shift) & 0xFFu)
         | ((c1 >> byte_shift) & 0xFFu) << 8
         | ((c2 >> byte_shift) & 0xFFu) << 16
         | ((c3 >> byte_shift) & 0xFFu) << 24;
}

inline float dot_group_device(const uint8_t * d, const uint8_t * qs,
                              const int8_t * activation, const int group) {
    const uint32_t packed = unpack_group(qs, group);
    const int8_t * a = activation + 8 * group;

    const uint32_t c0 = centroid_digits(index(packed, 0));
    const uint32_t c1 = centroid_digits(index(packed, 1));
    const uint32_t c2 = centroid_digits(index(packed, 2));
    const uint32_t c3 = centroid_digits(index(packed, 3));
    const uint32_t c4 = centroid_digits(index(packed, 4));
    const uint32_t c5 = centroid_digits(index(packed, 5));
    const uint32_t c6 = centroid_digits(index(packed, 6));
    const uint32_t c7 = centroid_digits(index(packed, 7));

    const int a_lo = int(pack_i8x4(a + 0));
    const int a_hi = int(pack_i8x4(a + 4));

    const int w0_lo = int(pack_digit_plane(c0, c1, c2, c3, 0));
    const int w0_hi = int(pack_digit_plane(c4, c5, c6, c7, 0));
    const int w1_lo = int(pack_digit_plane(c0, c1, c2, c3, 8));
    const int w1_hi = int(pack_digit_plane(c4, c5, c6, c7, 8));
    const int w2_lo = int(pack_digit_plane(c0, c1, c2, c3, 16));
    const int w2_hi = int(pack_digit_plane(c4, c5, c6, c7, 16));

    const int s0 = dpct::dp4a(w0_hi, a_hi, dpct::dp4a(w0_lo, a_lo, 0));
    const int s1 = dpct::dp4a(w1_hi, a_hi, dpct::dp4a(w1_lo, a_lo, 0));
    const int s2 = dpct::dp4a(w2_hi, a_hi, dpct::dp4a(w2_lo, a_lo, 0));

    // Reconstruct centroid*activation sum in FP32. Keeping the three planes
    // separate avoids int32 overflow from a 65536-weighted high plane.
    const float sum_micro = float(s0) + 256.0f * float(s1) + 65536.0f * float(s2);
    return scale(d[group]) * (sum_micro * 1.0e-6f);
}

#endif // __SYCL_DEVICE_ONLY__

inline float dot_group(const uint8_t * d, const uint8_t * qs, const int8_t * activation, const int group) {
#if defined(__SYCL_DEVICE_ONLY__)
    return dot_group_device(d, qs, activation, group);
#else
    const uint32_t packed = unpack_group(qs, group);
    const int8_t * a = activation + 8 * group;

    // Host/reference path deliberately retains direct exact-centroid FP32 math.
    float sum = centroid(index(packed, 0)) * float(a[0]);
    sum += centroid(index(packed, 1)) * float(a[1]);
    sum += centroid(index(packed, 2)) * float(a[2]);
    sum += centroid(index(packed, 3)) * float(a[3]);
    sum += centroid(index(packed, 4)) * float(a[4]);
    sum += centroid(index(packed, 5)) * float(a[5]);
    sum += centroid(index(packed, 6)) * float(a[6]);
    sum += centroid(index(packed, 7)) * float(a[7]);
    return scale(d[group]) * sum;
#endif
}

} // namespace ggml_sycl_tq3_4s

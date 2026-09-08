#!/usr/bin/env python3
"""Model-free numerical/source checks when a C++/SYCL toolchain is unavailable.

This simulates the lane schedule, not a SYCL runtime. The C++ tests separately
exercise the actual helpers, CPU decoder, and (in a SYCL build) GPU dispatch.
Only Python's standard library is required.
"""
from pathlib import Path
import math
import random
import re
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]
CPU = (ROOT / "ggml/src/ggml-quants.c").read_text(encoding="utf-8")
HELPERS = (ROOT / "ggml/src/ggml-sycl/tq3_4s.hpp").read_text(encoding="utf-8")
CUDA_DOT = (ROOT / "ggml/src/ggml-cuda/vecdotq.cuh").read_text(encoding="utf-8")


def array(source, name):
    body = re.search(r"\b" + name + r"\[[^]]+\]\s*=\s*\{([^}]+)\}", source).group(1)
    return [float(v.strip().rstrip("f")) for v in body.split(",") if v.strip()]


SIGNS = array(HELPERS, "signs")
CENTROIDS = array(HELPERS, "centroids")
REF_SIGNS = array(CPU, "TQ3_0_SIGNS")
REF_CENTROIDS = array(CPU, "TQ3_0_CENTROIDS")
NORM = float(re.search(r"rht_norm\s*=\s*([0-9.]+)f", HELPERS).group(1))


def f32(v):
    return struct.unpack("f", struct.pack("f", v))[0]


def f16(v):
    return struct.unpack("e", struct.pack("e", v))[0]


def cuda_codebook():
    words = [int(re.search(r"#define TQ3_4S_LEVELS_" + suffix + r"\s+0x([0-9A-Fa-f]+)u", CUDA_DOT).group(1), 16)
             for suffix in ("LO", "HI")]
    levels = struct.unpack("<8b", struct.pack("<II", *words))
    dot = CUDA_DOT[CUDA_DOT.index("float vec_dot_tq3_4s_q8_1("):]
    normalization = re.search(r"const float scale = \(([0-9.]+)f / ([0-9.]+)f\) \* ds8.x", dot)
    factor = float(normalization.group(1)) / float(normalization.group(2))
    return [level * factor for level in levels]


def scale(d):
    return 0.0 if d == 0 else ((32 + (d & 31)) * (1 << (d >> 5))) / 16384


def unpack(payload):
    result = []
    for group in range(4):
        p = payload[3 * group:3 * group + 3]
        packed = p[0] | p[1] << 8 | p[2] << 16
        result.extend((packed >> (3 * j)) & 7 for j in range(8))
    return result


def reference_unpack(payload):
    # Bit-by-bit, including indices crossing a byte boundary.
    return [sum(((payload[(3 * i + b) // 8] >> ((3 * i + b) % 8)) & 1) << b
                for b in range(3)) for i in range(32)]


def reference_rht(x, inverse=False):
    # Explicit orthonormal Hadamard matrix, natural order, no permutation.
    return [sum((-1 if (i & j).bit_count() % 2 else 1) * x[j] *
                (REF_SIGNS[i] if inverse else REF_SIGNS[j]) for j in range(32)) / math.sqrt(32)
            for i in range(32)]


def lane_rht(x, width):
    elements = 32 // width
    v = [[f32(x[lane * elements + j] * SIGNS[lane * elements + j])
          for j in range(elements)] for lane in range(width)]
    if elements == 2:
        v = [[f32(a + b), f32(a - b)] for a, b in v]
    step = elements
    while step < 32:
        old = [row[:] for row in v]
        for lane in range(width):
            for j in range(elements):
                other = old[lane ^ (step // elements)][j]
                v[lane][j] = f32(other - old[lane][j] if lane * elements & step else other + old[lane][j])
        step *= 2
    return [f32(value * f32(NORM)) for row in v for value in row]


def quantize(x, width):
    rotated = lane_rht(x, width)
    amax = max(map(abs, rotated))
    d = f32(amax / 127)
    qs = [0 if not amax else int(math.copysign(math.floor(abs(f32(v / d)) + 0.5), v)) for v in rotated]
    return qs, f16(d), d


class TQ3Math(unittest.TestCase):
    def assert_close(self, actual, expected, tolerance):
        self.assertTrue(math.isfinite(actual))
        self.assertLessEqual(abs(actual - expected), tolerance)

    def test_source_contract(self):
        self.assertEqual(SIGNS, REF_SIGNS)
        self.assertEqual(CENTROIDS, REF_CENTROIDS)
        cuda_signs = [-1 if ((i * 0x9E3779B9) & 0xFFFFFFFF) >> 31 else 1 for i in range(32)]
        self.assertEqual(SIGNS, cuda_signs)
        self.assertIn("tq3_0_rht_forward(x + i * QK_TQ3_0, rotated)",
                      CPU[CPU.index("void quantize_row_tq3_4s_ref"):CPU.index("void dequantize_row_tq3_4s")])
        backend = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text(encoding="utf-8")
        start = backend.index("static void ggml_sycl_mul_mat(")
        route = backend[start:backend.index("const bool split", start)]
        self.assertIn("GGML_TYPE_TQ3_4S", route)
        self.assertIn("ggml_sycl_op_mul_mat<quantize_tq3_4s_q8_1>", route)
        self.assertIn("is_gpu()", route)
        self.assertIn("return;", route)
        support = backend[backend.index("static bool do_ggml_backend_sycl_device_supports_op"):]
        support = support[support.index("case GGML_OP_MUL_MAT:"):support.index("case GGML_OP_OUT_PROD:")]
        self.assertIn("can_use_tq3_4s_mmvq(a, b, op)", support)
        self.assertIn("sub_group_sizes", support)
        self.assertIn("op->op == GGML_OP_MUL_MAT", support)
        self.assertIn("quantize_row_q8_1_sycl<quantize_tq3_4s_q8_1>", backend)
        mmvq = (ROOT / "ggml/src/ggml-sycl/mmvq.cpp").read_text(encoding="utf-8")
        dispatch = mmvq[mmvq.index("if (src0->type == GGML_TYPE_TQ3_4S)"):]
        dispatch = dispatch[:dispatch.index("return;")]
        self.assertIn("constexpr int tile_cols = 16;", dispatch)
        self.assertIn("first_col += tile_cols", dispatch)
        self.assertIn("mul_mat_vec_tq3_4s_q8_1_sycl_switch_ncols(", dispatch)
        kernel = mmvq[mmvq.index("static void mul_mat_vec_tq3_4s_q8_1_sycl"):]
        self.assertIn("[[sycl::reqd_sub_group_size(WARP_SIZE)]]", kernel)
        self.assertIn("ggml_sycl_tq3_4s::vdr, vec_dot_tq3_4s_q8_1>", kernel)

    def test_all_scales(self):
        for d in range(256):
            expected = math.ldexp(1 + (d & 31) / 32, (d >> 5) - 9) if d else 0
            self.assertEqual(scale(d), expected)
            self.assertEqual(f32(scale(d)), expected)

    def test_cuda_centroid_normalization_audit(self):
        # Audit the actual packed CUDA constants and the full 2.1519/127 factor.
        cuda = cuda_codebook()
        ratio_fn = CUDA_DOT[CUDA_DOT.index("float tq3_4s_ratio4s("):CUDA_DOT.index("#define TQ3_4S_LEVELS_LO")]
        exponent_bias = int(re.search(r"\+\s*(\d+)u\)\s*<<\s*23", ratio_fn).group(1))
        for d in range(256):
            bits = (((d >> 5) + exponent_bias) << 23) | ((d & 31) << 18)
            cuda_scale = 0.0 if d == 0 else struct.unpack("<f", struct.pack("<I", bits))[0]
            self.assertEqual(cuda_scale, scale(d))
        # The discrepancy is not removed by any shared scale normalization:
        # CUDA is antisymmetric, whereas the CPU codebook is not.
        for i in range(4):
            self.assertEqual(cuda[i] + cuda[7 - i], 0.0)
            self.assertGreater(abs(REF_CENTROIDS[i] + REF_CENTROIDS[7 - i]), 1e-3)
        # Concrete rotated-space dot counterexample: d=255, q8=1, d8=1.
        cpu_dot = scale(255) * REF_CENTROIDS[7]
        cuda_dot = scale(255) * cuda[7]
        self.assertGreater(abs(cuda_dot - cpu_dot), 0.08)
        self.assertGreater(max(abs(a - b) for a, b in zip(cuda, REF_CENTROIDS)), 0.16)

    def test_all_index_positions(self):
        for i in range(32):
            for value in range(8):
                payload = [0] * 12
                for b in range(3):
                    payload[(3 * i + b) // 8] |= ((value >> b) & 1) << ((3 * i + b) % 8)
                self.assertEqual(unpack(payload), reference_unpack(payload))
                self.assertEqual(unpack(payload)[i], value)
        for seed in (1, 42, 2026, 0xDEADBEEF):
            rng = random.Random(seed)
            for _ in range(1000):
                p = rng.randbytes(12)
                self.assertEqual(unpack(p), reference_unpack(p))

    def test_rht_basis_and_random(self):
        inputs = [[float(i == j) for i in range(32)] for j in range(32)]
        inputs += [[0.] * 32, [1.] * 32, [(-1.) ** i for i in range(32)]]
        for seed in (1, 42, 2026, 0xDEADBEEF):
            rng = random.Random(seed)
            inputs += [[f32(rng.gauss(0, 1)) for _ in range(32)] for _ in range(100)]
        for x in inputs:
            ref = reference_rht(x)
            for width in (16, 32):
                got = lane_rht(x, width)
                for a, b in zip(got, ref):
                    self.assert_close(a, b, 2e-6 * max(1, abs(b)))
                inv = reference_rht(got, inverse=True)
                for a, b in zip(inv, x):
                    self.assert_close(a, b, 2e-6 * max(1, abs(b)))
            self.assertEqual(lane_rht(x, 16), lane_rht(x, 32))

    def test_q8_scale_and_zero(self):
        self.assertEqual(quantize([0.] * 32, 16), ([0] * 32, 0., 0.))
        for amplitude in (1e-8, 1e-4, 1., 100.):
            rng = random.Random(42)
            for _ in range(100):
                x = [f32(amplitude * rng.gauss(0, 1)) for _ in range(32)]
                q, d, raw_d = quantize(x, 16)
                self.assertEqual((q, d, raw_d), quantize(x, 32))
                for actual, integer in zip(reference_rht(x), q):
                    self.assertLessEqual(abs(integer), 127)
                    self.assert_close(integer * d, actual,
                                      raw_d / 2 + 127 * abs(raw_d - d) + 2e-6 * amplitude)

    def test_dot_and_matvec(self):
        # Four MMVQ lanes per 32 weights, qi=8, vdr=2; padded activation rows.
        for seed in (1, 42, 2026, 0xDEADBEEF):
            rng = random.Random(seed)
            for k in (32, 64, 96, 256, 512, 544, 1024, 4096):
                blocks = k // 32
                rows, cols = 3, 3
                pitch = ((k + 511) // 512) * 512 // 32
                weights = [[(list(rng.randbytes(4)), list(rng.randbytes(12))) for _ in range(blocks)] for _ in range(rows)]
                activations = [[f32(rng.gauss(0, 1)) for _ in range(k)] for _ in range(cols)]
                activations[1] = [0.] * k
                for width in (16, 32):
                    quantized = [None] * (cols * pitch)
                    for c, x in enumerate(activations):
                        for b in range(blocks):
                            quantized[c * pitch + b] = quantize(x[32 * b:32 * b + 32], width)
                    for row in range(rows):
                        for c, x in enumerate(activations):
                            sums = [0.] * width
                            original = expected = bound = magnitude = 0.
                            for b, (d, packed) in enumerate(weights[row]):
                                indices = reference_unpack(packed)
                                rotated_w = [REF_CENTROIDS[i] * scale(d[j // 8]) for j, i in enumerate(indices)]
                                logical_w = reference_rht(rotated_w, inverse=True)
                                qs, qd, raw_d = quantized[c * pitch + b]
                                original += sum(w * a for w, a in zip(logical_w, x[32 * b:32 * b + 32]))
                                expected += sum(w * q * qd for w, q in zip(rotated_w, qs))
                                magnitude += sum(abs(w * q * qd) for w, q in zip(rotated_w, qs))
                                bound += sum(map(abs, rotated_w)) * (raw_d / 2 + 127 * abs(raw_d - qd))
                            # Actual generic MMVQ lane/block assignment, including tail blocks.
                            for lane in range(width):
                                group = lane % 4
                                for b in range(lane // 4, blocks, width // 4):
                                    d, packed = weights[row][b]
                                    indices = unpack(packed)
                                    qs, qd, _ = quantized[c * pitch + b]
                                    dot = 0.
                                    for j in range(8):
                                        i = 8 * group + j
                                        dot = f32(dot + f32(f32(CENTROIDS[indices[i]]) * qs[i]))
                                    sums[lane] = f32(sums[lane] + f32(f32(dot * scale(d[group])) * qd))
                            mask = width // 2
                            while mask:
                                old = sums[:]
                                sums = [f32(old[lane] + old[lane ^ mask]) for lane in range(width)]
                                mask //= 2
                            tolerance = 2e-5 * max(1, magnitude)
                            self.assert_close(sums[0], expected, tolerance)
                            self.assert_close(sums[0], original, bound + tolerance)


if __name__ == "__main__":
    cuda_signs = [-1 if ((i * 0x9E3779B9) & 0xFFFFFFFF) >> 31 else 1 for i in range(32)]
    differences = [i for i in range(32) if cuda_signs[i] != REF_SIGNS[i]]
    print(f"CPU vs CUDA RHT sign differences (zero-based): {differences}", flush=True)
    print("CUDA centroids after full scale normalization:", [round(c, 9) for c in cuda_codebook()], flush=True)
    print("CPU encoder/decoder centroids:", REF_CENTROIDS, flush=True)
    print("These are numerical/source checks, not compiled C++ or GPU execution.", flush=True)
    unittest.main(verbosity=2)

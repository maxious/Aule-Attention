#!/usr/bin/env python3
import sys
import os
import ctypes
import numpy as np
import time
from datetime import datetime


def load_library():
    lib_paths = [
        "../zig-out/lib/libaule.so",
        "./zig-out/lib/libaule.so",
    ]
    for path in lib_paths:
        if os.path.exists(path):
            print(f"✓ Loading library from: {path}")
            return ctypes.CDLL(path)
    raise FileNotFoundError("Could not find libaule.so - run 'zig build' first")


def setup_library(lib):
    lib.aule_init.restype = ctypes.c_int32
    lib.aule_shutdown.restype = None
    lib.aule_get_error.restype = ctypes.c_char_p
    lib.aule_get_vendor.restype = ctypes.c_int32
    lib.aule_get_device_name.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
    lib.aule_get_device_name.restype = ctypes.c_int32

    lib.aule_tensor_create.argtypes = [ctypes.c_uint32] * 4
    lib.aule_tensor_create.restype = ctypes.c_uint64
    lib.aule_tensor_destroy.argtypes = [ctypes.c_uint64]
    lib.aule_tensor_upload.argtypes = [
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
    ]
    lib.aule_tensor_upload.restype = ctypes.c_int32
    lib.aule_tensor_download.argtypes = [
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
    ]
    lib.aule_tensor_download.restype = ctypes.c_int32

    lib.aule_tensor_create_f16.argtypes = [ctypes.c_uint32] * 4
    lib.aule_tensor_create_f16.restype = ctypes.c_uint64
    lib.aule_tensor_create_bf16.argtypes = [ctypes.c_uint32] * 4
    lib.aule_tensor_create_bf16.restype = ctypes.c_uint64

    lib.aule_tensor_upload_u16.argtypes = [
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_uint32,
    ]
    lib.aule_tensor_upload_u16.restype = ctypes.c_int32

    lib.aule_tensor_download_u16.argtypes = [
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_uint32,
    ]
    lib.aule_tensor_download_u16.restype = ctypes.c_int32

    lib.aule_set_shader_variant.argtypes = [ctypes.c_uint8]
    lib.aule_set_shader_variant.restype = ctypes.c_int32

    lib.aule_has_shader_variant.argtypes = [ctypes.c_uint8]
    lib.aule_has_shader_variant.restype = ctypes.c_int32

    lib.aule_attention_forward_gpu.argtypes = (
        [ctypes.c_uint64] * 4 + [ctypes.c_uint64] * 2 + [ctypes.c_int32, ctypes.c_int32]
    )
    lib.aule_attention_forward_gpu.restype = ctypes.c_int32


def get_device_info(lib):
    vendor = lib.aule_get_vendor()
    vendor_names = {0: "Other", 1: "AMD", 2: "NVIDIA", 3: "Intel", 4: "Apple"}

    device_name = ctypes.create_string_buffer(256)
    lib.aule_get_device_name(device_name, 256)

    # Check shader support
    bf16_supported = lib.aule_has_shader_variant(4)  # 4 = bf16

    return {
        "vendor": vendor_names.get(vendor, "Unknown"),
        "vendor_id": vendor,
        "device": device_name.value.decode(),
        "bf16_supported": bf16_supported == 1,
    }


def calculate_flops(batch, heads, seq_len, head_dim, time_ms):
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    tflops = (flops / (time_ms / 1000)) / 1e12
    return tflops


def float_to_float16(arr):
    return arr.astype(np.float16).view(np.uint16)


def float_to_bfloat16(arr):
    # Quick and dirty conversion for testing
    u32_view = arr.view(np.uint32)
    return (u32_view >> 16).astype(np.uint16)


def benchmark_config(
    lib, name, batch, heads, seq_len, head_dim, dtype="fp32", warmup=3, iterations=10
):
    print(f"\n--- {name} ({dtype}) ---")
    print(
        f"Config: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}"
    )

    count = batch * heads * seq_len * head_dim

    # Select shader variant and tensor creation
    variant_map = {
        "fp32": 0,  # baseline
        "fp32_fast": 1,
        "fp16": 2,
        "fp16_amd": 3,
        "bf16": 4,
    }

    variant = variant_map.get(dtype, 0)

    if lib.aule_has_shader_variant(variant) != 1:
        print(f"✗ Shader variant {dtype} not available/supported")
        return None

    lib.aule_set_shader_variant(variant)

    if dtype == "bf16":
        create_fn = lib.aule_tensor_create_bf16
        upload_fn = lib.aule_tensor_upload_u16
        # Generate data and convert to bf16
        data_f32 = np.random.randn(count).astype(np.float32) * 0.02
        data_u16 = float_to_bfloat16(data_f32)
        upload_ptr = data_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    elif dtype.startswith("fp16"):
        create_fn = lib.aule_tensor_create_f16
        upload_fn = lib.aule_tensor_upload_u16
        # Generate data and convert to fp16
        data_f32 = np.random.randn(count).astype(np.float32) * 0.02
        data_u16 = float_to_float16(data_f32)
        upload_ptr = data_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    else:
        create_fn = lib.aule_tensor_create
        upload_fn = lib.aule_tensor_upload
        data_f32 = np.random.randn(count).astype(np.float32) * 0.02
        upload_ptr = data_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    Q = create_fn(batch, heads, seq_len, head_dim)
    K = create_fn(batch, heads, seq_len, head_dim)
    V = create_fn(batch, heads, seq_len, head_dim)
    O = create_fn(batch, heads, seq_len, head_dim)

    upload_fn(Q, upload_ptr, count)
    upload_fn(K, upload_ptr, count)
    upload_fn(V, upload_ptr, count)

    forward_fn = lib.aule_attention_forward_gpu

    for _ in range(warmup):
        forward_fn(Q, K, V, O, 0, 0, 1, -1)

    times = []
    for _ in range(iterations):
        start = time.time()
        ret = forward_fn(Q, K, V, O, 0, 0, 1, -1)

        if hasattr(lib, "aule_wait_idle"):
            lib.aule_wait_idle()

        elapsed = (time.time() - start) * 1000

        if ret != 0:
            error = lib.aule_get_error()
            print(f"✗ Error: {ctypes.string_at(error).decode()}")
            break

        times.append(elapsed)

    lib.aule_tensor_destroy(Q)
    lib.aule_tensor_destroy(K)
    lib.aule_tensor_destroy(V)
    lib.aule_tensor_destroy(O)

    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        tflops = calculate_flops(batch, heads, seq_len, head_dim, avg_time)
        throughput = (batch * seq_len) / (avg_time / 1000)

        return {
            "name": name,
            "batch": batch,
            "heads": heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "avg_time_ms": avg_time,
            "std_time_ms": std_time,
            "tflops": tflops,
            "throughput": throughput,
            "success": True,
        }

    return None


def print_results_table(results):
    print("\n" + "=" * 130)
    print("BENCHMARK RESULTS (head_dim=64)")
    print("=" * 130)
    print(
        f"{'Config':<35} {'DType':<10} {'Batch':<6} {'Heads':<6} {'SeqLen':<7} {'Time(ms)':<12} {'TFLOPS':<10} {'Throughput':<15} {'Status':<8}"
    )
    print("-" * 130)

    for r in results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        throughput_str = f"{r['throughput']:.0f} tok/s"
        dtype = r.get("dtype", "fp32")
        print(
            f"{r['name']:<35} {dtype:<10} {r['batch']:<6} {r['heads']:<6} {r['seq_len']:<7} "
            f"{r['avg_time_ms']:>8.2f}±{r['std_time_ms']:.2f} {r['tflops']:>8.2f} "
            f"{throughput_str:<15} {status:<8}"
        )

    print("=" * 130)


def main():
    print("=" * 80)
    print("  Aule Attention Vulkan Benchmark Suite")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    try:
        lib = load_library()
        setup_library(lib)

        ret = lib.aule_init()
        if ret != 0:
            error = lib.aule_get_error()
            print(f"✗ Init failed: {ctypes.string_at(error).decode()}")
            return False

        device_info = get_device_info(lib)
        print(f"\n✓ GPU: {device_info['device']}")
        print(f"  Vendor: {device_info['vendor']}")
        print(f"  BF16 Supported: {device_info['bf16_supported']}")

        # Force attention engine initialization by doing a small attention call
        print("  Initializing attention engine...")
        Q = lib.aule_tensor_create(1, 1, 1, 64)
        K = lib.aule_tensor_create(1, 1, 1, 64)
        V = lib.aule_tensor_create(1, 1, 1, 64)
        O = lib.aule_tensor_create(1, 1, 1, 64)

        data = np.ones(64, dtype=np.float32)
        lib.aule_tensor_upload(
            Q, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 64
        )
        lib.aule_tensor_upload(
            K, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 64
        )
        lib.aule_tensor_upload(
            V, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 64
        )

        # This should trigger engine initialization
        result = lib.aule_attention_forward_gpu(Q, K, V, O, 0, 0, 0, -1)
        if result != 0:
            error = lib.aule_get_error()
            print(
                f"  Warning: Engine init test failed: {ctypes.string_at(error).decode()}"
            )

        # Cleanup
        lib.aule_tensor_destroy(Q)
        lib.aule_tensor_destroy(K)
        lib.aule_tensor_destroy(V)
        lib.aule_tensor_destroy(O)

        print("\n" + "=" * 80)
        print("  Running benchmarks...")
        print("=" * 80)

        results = []

        configs = [
            ("Small Decode (1 token)", 1, 32, 1, 64),
            ("Medium Prefill (256)", 1, 32, 256, 64),
            ("Large Context (2K)", 1, 32, 2048, 64),
            # ("Very Large (4K)", 1, 32, 4096, 64),
            ("Batched Small (B=8)", 8, 32, 128, 64),
        ]

        # Test available variants
        variants_to_test = ["fp32"]
        if lib.aule_has_fp16() == 1:
            variants_to_test.append("fp16")
            if lib.aule_is_amd_optimized() == 1:
                variants_to_test.append("fp16_amd")

        # Check for bf16 support
        if lib.aule_has_shader_variant(4) == 1:
            variants_to_test.append("bf16")

        print(f"Testing variants: {variants_to_test}")

        for dtype in variants_to_test:
            for name, batch, heads, seq_len, head_dim in configs:
                result = benchmark_config(
                    lib,
                    name,
                    batch,
                    heads,
                    seq_len,
                    head_dim,
                    dtype=dtype,
                    warmup=3,
                    iterations=10,
                )
                if result:
                    result["dtype"] = dtype
                    results.append(result)
                    print(
                        f"  ✓ {name} ({dtype}): {result['avg_time_ms']:.2f}ms, {result['tflops']:.2f} TFLOPS"
                    )

        print_results_table(results)

        lib.aule_shutdown()
        return True

    except Exception as e:
        print(f"✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

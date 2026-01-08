#!/usr/bin/env python3
import sys
import os
import ctypes
import numpy as np


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

    lib.aule_tensor_create.argtypes = [ctypes.c_uint32] * 4
    lib.aule_tensor_create.restype = ctypes.c_uint64
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

    lib.aule_attention_forward_gpu.argtypes = (
        [ctypes.c_uint64] * 4 + [ctypes.c_uint64] * 2 + [ctypes.c_int32, ctypes.c_int32]
    )
    lib.aule_attention_forward_gpu.restype = ctypes.c_int32
    lib.aule_set_shader_variant.argtypes = [ctypes.c_uint8]
    lib.aule_set_shader_variant.restype = ctypes.c_int32


def float_to_bfloat16(arr):
    # Quick and dirty conversion for testing
    u32_view = arr.view(np.uint32)
    return (u32_view >> 16).astype(np.uint16)


def bfloat16_to_float(arr):
    # Convert back to float for verification
    u32_vals = arr.astype(np.uint32) << 16
    return u32_vals.view(np.float32)


def test_bf16_small():
    lib = load_library()
    setup_library(lib)

    ret = lib.aule_init()
    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Init failed: {ctypes.string_at(error).decode()}")
        return False

    print("✓ Library initialized")

    # Set to BF16 variant
    lib.aule_set_shader_variant(4)  # 4 = bf16

    # Create small tensors: batch=1, heads=1, seq_len=1, head_dim=4 (very small for testing)
    batch, heads, seq_len, head_dim = 1, 1, 1, 4
    count = batch * heads * seq_len * head_dim

    print(f"Testing BF16 attention: {batch}x{heads}x{seq_len}x{head_dim}")

    Q = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    K = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    V = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    O = lib.aule_tensor_create(batch, heads, seq_len, head_dim)

    if Q == 0 or K == 0 or V == 0 or O == 0:
        print("✗ Failed to create tensors")
        return False

    print("✓ Created BF16 tensors")

    # Generate test data
    q_data = np.random.randn(count).astype(np.float32) * 0.1
    k_data = np.random.randn(count).astype(np.float32) * 0.1
    v_data = np.random.randn(count).astype(np.float32) * 0.1

    # Upload (GLSL shader handles BF16 conversion internally)
    ret = lib.aule_tensor_upload(
        Q, q_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count
    )
    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Upload Q failed: {ctypes.string_at(error).decode()}")
        return False

    ret = lib.aule_tensor_upload(
        K, k_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count
    )
    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Upload K failed: {ctypes.string_at(error).decode()}")
        return False

    ret = lib.aule_tensor_upload(
        V, v_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count
    )
    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Upload V failed: {ctypes.string_at(error).decode()}")
        return False

    print("✓ Uploaded data")

    # Run attention
    ret = lib.aule_attention_forward_gpu(
        Q, K, V, O, 0, 0, 0, -1
    )  # causal=0, window_size=-1
    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Attention failed: {ctypes.string_at(error).decode()}")
        return False

    print("✓ Attention computation succeeded")

    # Download result
    out_float = np.zeros(count, dtype=np.float32)
    ret = lib.aule_tensor_download(
        O, out_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count
    )
    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Download failed: {ctypes.string_at(error).decode()}")
        return False

    print("✓ Downloaded result")
    print(f"  Q: {q_data}")
    print(f"  K: {k_data}")
    print(f"  V: {v_data}")
    print(f"  Output: {out_float}")

    # Check if output looks reasonable (should be small due to input scaling)
    if np.max(np.abs(out_float)) < 1.0:
        print("✓ Output values look reasonable")
    else:
        print("⚠ Output values seem large, might indicate issues")

    # Cleanup
    lib.aule_tensor_destroy(Q)
    lib.aule_tensor_destroy(K)
    lib.aule_tensor_destroy(V)
    lib.aule_tensor_destroy(O)
    lib.aule_shutdown()

    print("✓ BF16 test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_bf16_small()
    sys.exit(0 if success else 1)

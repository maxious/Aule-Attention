# BFloat16 Support in Vulkan Shaders

## Current Status (January 2026)

### The Problem

**Neither GLSL nor Slang have native bfloat16 types yet.**

While Vulkan has `VK_KHR_shader_bfloat16`, high-level language support is missing.

### Solution: Emulation

We treat `bfloat16` data as `uint16_t` (packed into `uint32_t` pairs) in shaders and perform software conversion to `float` for arithmetic operations.

This allows us to support `bfloat16` IO and computation on any GPU, regardless of native driver support.

### Approach: Emulated BF16

This project uses **emulated bf16 support via GLSL**. 

We use `uint` buffers to store bf16 data and perform unpacking/packing in the shader. 
This approach works on ALL Vulkan 1.2+ hardware, including those that do not support `VK_KHR_shader_bfloat16`.

## Hardware Support

Since we use emulation, **bf16 mode works on any GPU** that supports Vulkan 1.2 (StorageBuffer support).
- **NVIDIA**: Supported
- **AMD**: Supported
- **Intel**: Supported
- **Apple**: Supported via MoltenVK

No `VK_KHR_shader_bfloat16` extension is required.

## Current Implementation

- `shaders/attention_bf16.spv` - Emulated bf16 attention shader (compiled from GLSL)

### Building

The shader is pre-compiled and included in the repository.

### No Fallback Needed

Because the implementation is emulated, it serves as the universal fallback.

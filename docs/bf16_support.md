# BFloat16 Support in Vulkan Shaders

## Current Status (January 2026)

### The Problem

**Neither GLSL nor Slang have native bfloat16 types yet.**

While Vulkan has `VK_KHR_shader_bfloat16` (approved March 2025) and SPIR-V has `SPV_KHR_bfloat16`, 
there is **no corresponding high-level shading language support**:

```glsl
// GLSL - DOES NOT WORK
#extension GL_KHR_shader_bfloat16 : require  // Extension does not exist
bf16vec4 myValue;  // Type not available
```

```slang
// Slang - DOES NOT WORK (as of v2025.24)
StructuredBuffer<bfloat16> buffer;  // Type not recognized
```

**Neither glslc nor slangc can compile native bf16 shaders.**

### What VK_KHR_shader_bfloat16 Provides

The extension adds SPIR-V capabilities only:
- `BFloat16TypeKHR` - Basic bf16 type declaration and conversions
- `BFloat16DotProductKHR` - bf16 dot products via `OpDot`
- `BFloat16CooperativeMatrixKHR` - bf16 cooperative matrices

These are accessible only through:
1. **Hand-written SPIR-V assembly** (tested and working, see `shaders/test_bf16_native.spvasm`)
2. Custom shader compilers (like NCNN's extended GLSL preprocessor)
3. Future GLSL/Slang extensions (when available)

### Approach: Native BF16 Only

This project uses **native bf16 via hand-written SPIR-V assembly**. If hardware doesn't support `VK_KHR_shader_bfloat16`, bf16 mode is refused at runtime.

No emulation fallback is provided - use fp16 or fp32 instead on unsupported hardware.

## Hardware Support

`VK_KHR_shader_bfloat16` requires GPU support:
- **NVIDIA**: Ada Lovelace (RTX 40xx) and newer
- **AMD**: RDNA3 and newer (limited support)
- **Intel**: Arc GPUs

Query with:
```c
VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16Features = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR
};
```

## Current Implementation

This project now supports **native bf16 via hand-written SPIR-V assembly**:

- `shaders/attention_bf16_native.spvasm` - Native bf16 attention dot product
- `shaders/test_bf16_native.spvasm` - Simple bf16 test shader

### Building

```bash
# If spirv-as is not in PATH, specify location:
zig build -Dspirv-tools=$HOME/vulkan/1.4.335.0/x86_64/bin
```

### Hardware Requirements

The native bf16 shaders require:
- `VK_KHR_shader_bfloat16` extension
- GPU with bf16 support (NVIDIA Ada+, AMD RDNA3+, Intel Arc)

### No Fallback

BF16 mode requires hardware support. On unsupported GPUs, the runtime will refuse to create bf16 pipelines - use fp16 or fp32 modes instead.

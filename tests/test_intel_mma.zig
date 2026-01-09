const std = @import("std");
const AttentionEngine = @import("aule").AttentionContext;
const GpuTensor = @import("aule").DType;

const log = std.log.scoped(.test_intel_mma);

fn floatToBf16(f: f32) u16 {
    const bits = @as(u32, @bitCast(f));
    return @intCast(bits >> 16);
}

fn bf16ToFloat(u: u16) f32 {
    const bits = @as(u32, u) << 16;
    return @bitCast(bits);
}

test "Intel XMX/MMA BF16: correctness test" {
    const allocator = std.testing.allocator;

    // Load shaders
    const attention_f32_spv = @embedFile("attention_f32_spv");
    const attention_amd_spv = @embedFile("attention_amd_spv");
    const attention_paged_spv = @embedFile("attention_paged_spv");
    const copy_kv_spv = @embedFile("copy_kv_to_paged_spv");
    const attention_bf16_intel_mma_spv = @embedFile("attention_bf16_intel_mma_spv");
    const attention_bf16_spv = @embedFile("attention_bf16_spv");

    // Initialize Engine
    var engine = AttentionEngine.initWithBackward(
        allocator,
        attention_f32_spv,
        attention_amd_spv,
        null, // forward_lse
        null, // backward
        null, // sort
        null, // gravity
        null, // radix_count
        null, // radix_scan
        null, // radix_scatter
        null, // iota
        null, // magnitude
        null, // fast
        null, // fp16
        null, // fp16_amd
        attention_bf16_spv,
        null,
        attention_bf16_intel_mma_spv,
        null,
        attention_paged_spv,
        copy_kv_spv,
    ) catch |err| {
        log.err("Failed to initialize AttentionEngine: {}", .{err});
        return err;
    };
    defer engine.deinit();

    if (engine.backend != .vulkan) {
        log.warn("Skipping Intel MMA test: Backend is {s} (requires Vulkan). Please ensure Vulkan drivers are installed and a GPU is available.", .{engine.getBackendName()});
        return;
    }

    // Verify Intel XMX/MMA support
    var has_intel_mma = false;
    if (engine.vulkan_ctx) |vctx| {
        log.info("Selected Device: {s}", .{vctx.ctx.gpu_caps.getDeviceName()});
        has_intel_mma = (vctx.ctx.gpu_caps.vendor == .intel);
        if (has_intel_mma and !vctx.ctx.gpu_caps.hasIntelMMA()) {
            log.warn("Intel GPU detected but required extensions missing.", .{});
        }
    }

    if (!has_intel_mma) {
        log.warn("Intel XMX/MMA Hardware not detected (Vendor={?})", .{if (engine.vulkan_ctx) |v| v.ctx.gpu_caps.vendor else null});
        log.warn("Forcing execution as requested...", .{});
        // return; // FORCE RUN
    } else {
        log.info("Intel Hardware detected, proceeding with MMA correctness test...", .{});
    }

    // Configuration

    const batch_size: u32 = 1;
    const num_heads: u32 = 1;
    const seq_len: u32 = 1; // M=1
    const key_seq_len: u32 = 8; // N=8
    const head_dim: u32 = 16; // K=16

    const shape_q = [4]u32{ batch_size, num_heads, seq_len, head_dim };
    const shape_k = [4]u32{ batch_size, num_heads, key_seq_len, head_dim };
    const shape_v = [4]u32{ batch_size, num_heads, key_seq_len, head_dim };
    const shape_o = [4]u32{ batch_size, num_heads, seq_len, head_dim };

    // Create tensors with BF16 dtype
    var Q = try engine.createTensor(shape_q, .bf16);
    defer engine.destroyTensor(&Q);

    var K = try engine.createTensor(shape_k, .bf16);
    defer engine.destroyTensor(&K);

    var V = try engine.createTensor(shape_v, .bf16);
    defer engine.destroyTensor(&V);

    var output = try engine.createTensor(shape_o, .bf16);
    defer engine.destroyTensor(&output);

    // Prepare Host Data
    const q_data = try allocator.alloc(u16, 16);
    defer allocator.free(q_data);
    @memset(q_data, floatToBf16(1.0));

    const k_data = try allocator.alloc(u16, 8 * 16);
    defer allocator.free(k_data);

    for (0..8) |i| {
        const val: f32 = switch (i) {
            0 => 1.0,
            1 => 0.5,
            2 => 2.0,
            else => 0.0,
        };
        const bf_val = floatToBf16(val);
        for (0..16) |j| {
            k_data[i * 16 + j] = bf_val;
        }
    }

    const v_data = try allocator.alloc(u16, 8 * 16);
    defer allocator.free(v_data);
    @memset(v_data, floatToBf16(0.0));

    // Upload
    try engine.upload_u16(&Q, q_data);
    try engine.upload_u16(&K, k_data);
    try engine.upload_u16(&V, v_data);

    // Set Shader Variant
    if (engine.vulkan_ctx) |vctx| {
        // Force the Intel MMA pipeline
        try vctx.setShaderVariant(.intel_mma_bf16);
    }

    log.info("Dispatching Intel MMA kernel...", .{});

    // Dispatch
    try engine.attention(&Q, &K, &V, &output, null, null, false, -1);

    // Synchronize via vulkan_ctx directly
    if (engine.vulkan_ctx) |vctx| {
        try vctx.synchronize();
    }

    // Download Results
    const out_data = try allocator.alloc(u16, 1 * 1 * 1 * 16);
    defer allocator.free(out_data);
    try engine.download_u16(&output, out_data);

    // Verify
    log.info("Verification Results:", .{});

    const expected = [_]f32{ 16.0, 8.0, 32.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    var success = true;
    for (0..8) |i| {
        const val = bf16ToFloat(out_data[i]);
        const exp = expected[i];
        const diff = @abs(val - exp);
        const tolerance = 0.5;

        log.info("  Lane {d}: Expected {d:.2}, Got {d:.2}", .{ i, exp, val });

        if (diff > tolerance) {
            log.err("Mismatch at lane {d}!", .{i});
            success = false;
        }
    }

    try std.testing.expect(success);
}

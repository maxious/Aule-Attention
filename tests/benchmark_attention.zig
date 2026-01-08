const std = @import("std");
const aule = @import("aule");
const Attention = aule.Attention;
const ShaderVariant = aule.ShaderVariant;
const DType = aule.DType;

pub fn main() !void {
    // Setup allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize Engine
    std.debug.print("Initializing aule-attention...\n", .{});
    var attn = try Attention.init(allocator);
    defer attn.deinit();
    const ctx = &attn.context;

    // Configuration
    const batch = 4;
    const heads = 8;
    const seq = 1024;
    const dim = 64;

    std.debug.print("Benchmarking config: B={} H={} S={} D={}\n", .{ batch, heads, seq, dim });

    // 1. Benchmark FP32 (Fast)
    try runBenchmark(ctx, allocator, batch, heads, seq, dim, .f32, .fast, "FP32 (Fast)");

    // 2. Benchmark FP32 (Baseline)
    try runBenchmark(ctx, allocator, batch, heads, seq, dim, .f32, .baseline, "FP32 (Baseline)");

    // 3. Benchmark FP32 (Cooperative Matrix)
    try runBenchmark(ctx, allocator, batch, heads, seq, dim, .f32, .coopmat, "FP32 (Cooperative Matrix)");

    // 4. Benchmark BF16
    try runBenchmark(ctx, allocator, batch, heads, seq, dim, .bf16, .bf16, "BF16 (Emulated)");
}

fn runBenchmark(ctx: *aule.AttentionContext, allocator: std.mem.Allocator, batch: u32, heads: u32, seq: u32, dim: u32, dtype: DType, variant: ShaderVariant, name: []const u8) !void {
    std.debug.print("\n=== Benchmarking {s} ===\n", .{name});

    const shape = [4]u32{ batch, heads, seq, dim };
    const total_elements = batch * heads * seq * dim;

    // Set Shader Variant
    if (ctx.vulkan_ctx) |engine| {
        try engine.setShaderVariant(variant);
    } else {
        std.debug.print("Not using Vulkan backend, skipping variant set\n", .{});
        return;
    }

    // Allocate GPU tensors
    var q_t = try ctx.createTensor(shape, dtype);
    defer ctx.destroyTensor(&q_t);
    var k_t = try ctx.createTensor(shape, dtype);
    defer ctx.destroyTensor(&k_t);
    var v_t = try ctx.createTensor(shape, dtype);
    defer ctx.destroyTensor(&v_t);
    var o_t = try ctx.createTensor(shape, dtype);
    defer ctx.destroyTensor(&o_t);

    // Upload dummy data
    if (dtype == .f32) {
        const host_data = try allocator.alloc(f32, total_elements);
        defer allocator.free(host_data);
        @memset(host_data, 0.1);
        try ctx.upload(&q_t, host_data);
        try ctx.upload(&k_t, host_data);
        try ctx.upload(&v_t, host_data);
    } else {
        // For BF16/FP16, we upload u16 data
        const host_data = try allocator.alloc(u16, total_elements);
        defer allocator.free(host_data);
        @memset(host_data, 0x3F80); // 1.0 in bf16 (roughly) or fp16, doesn't matter for perf
        try ctx.upload_u16(&q_t, host_data);
        try ctx.upload_u16(&k_t, host_data);
        try ctx.upload_u16(&v_t, host_data);
    }

    // Warmup
    std.debug.print("Warming up...\n", .{});
    try ctx.attention(&q_t, &k_t, &v_t, &o_t, null, null, false, -1);
    try ctx.vulkan_ctx.?.synchronize();

    // Benchmark Loop
    const iterations = 50;
    var timer = try std.time.Timer.start();

    std.debug.print("Running {} iterations...\n", .{iterations});
    const start = timer.read();
    for (0..iterations) |_| {
        try ctx.attention(&q_t, &k_t, &v_t, &o_t, null, null, false, -1);
    }
    // Wait for idle at the end
    try ctx.vulkan_ctx.?.synchronize();
    const end = timer.read();

    const total_ns = end - start;
    const avg_ms = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iterations)) / 1_000_000.0;

    // Calculate TFLOPS
    // Ops = 4 * B * H * S^2 * D
    const ops_per_iter = 4.0 * @as(f64, @floatFromInt(batch)) *
        @as(f64, @floatFromInt(heads)) *
        @as(f64, @floatFromInt(seq)) *
        @as(f64, @floatFromInt(seq)) *
        @as(f64, @floatFromInt(dim));

    const tflops = (ops_per_iter) / (avg_ms / 1000.0) / 1_000_000_000_000.0;

    std.debug.print("--------------------------------------------------\n", .{});
    std.debug.print("Average Time: {d:.3} ms\n", .{avg_ms});
    std.debug.print("Throughput:   {d:.3} TFLOPS\n", .{tflops});
    std.debug.print("--------------------------------------------------\n", .{});
}

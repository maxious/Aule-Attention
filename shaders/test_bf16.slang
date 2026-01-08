// BFloat16 test shader using Slang
// Requires VK_KHR_shader_bfloat16 extension

[[vk::binding(0, 0)]]
StructuredBuffer<bfloat16> inputA;

[[vk::binding(1, 0)]]
StructuredBuffer<bfloat16> inputB;

[[vk::binding(2, 0)]]
RWStructuredBuffer<bfloat16> output;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    
    // Load bf16 values
    bfloat16 a = inputA[index];
    bfloat16 b = inputB[index];
    
    // Perform bf16 arithmetic
    bfloat16 result = a + b;
    
    // Store result
    output[index] = result;
}

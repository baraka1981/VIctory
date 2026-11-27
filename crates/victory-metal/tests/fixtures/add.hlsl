// victory-metal test kernel: out[i] = input[i] * 2 + base.
// Root signature = victory_host::ComputeDesc's model:
// param 0 constants, then one root SRV / UAV per binding.
// Build: dxc -T cs_6_0 -E main add.hlsl -Fo add.dxil
#define RS "RootConstants(num32BitConstants=1, b0), SRV(t0), UAV(u0)"
cbuffer Constants : register(b0) { uint base; };
StructuredBuffer<uint> input : register(t0);
RWStructuredBuffer<uint> output : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { output[id.x] = input[id.x] * 2 + base; }

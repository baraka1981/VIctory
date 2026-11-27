// victory-metal test kernel: out[i] = input[i] * scale + base, with the
// input, output and scale reached through one descriptor table (SRV, UAV,
// CBV in a view heap) and base a root constant. Root signature embedded.
//   dxc -T cs_6_0 -E main table.hlsl -Fo table.dxil
#define RS "RootConstants(num32BitConstants=1, b0), DescriptorTable(SRV(t0), UAV(u0), CBV(b1))"
cbuffer Root : register(b0) { uint base; };
cbuffer Table : register(b1) { uint scale; };
StructuredBuffer<uint> input : register(t0);
RWStructuredBuffer<uint> output : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { output[id.x] = input[id.x] * scale + base; }

// victory-metal test kernel: out[i] = input[i] * scale + base, with the
// root signature handed in separately (ComputeDesc::root_signature), in an
// order unlike add.hlsl's: UAV address first, then constants, then SRV.
// The DXIL is built with the root signature stripped, so only the one
// handed in exists (victory-metal's toplevel::swap_root_signature makes it):
//   dxc -T cs_6_0 -E main -Qstrip_rootsignature swap.hlsl -Fo swap.dxil
#define RS "UAV(u0), RootConstants(num32BitConstants=2, b0), SRV(t0, flags=DATA_STATIC)"
cbuffer Constants : register(b0) { uint scale; uint base; };
StructuredBuffer<uint> input : register(t0);
RWStructuredBuffer<uint> output : register(u0);
[RootSignature(RS)]
[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { output[id.x] = input[id.x] * scale + base; }

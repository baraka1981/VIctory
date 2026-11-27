// victory-metal test pixel shader: the colour in root constants b0.
// Used with quad_vs.hlsl and mesh.hlsl.
//   dxc -T ps_6_0 -E main quad_ps.hlsl -Fo quad_ps.dxil
#define RS "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), RootConstants(num32BitConstants=4, b0)"
cbuffer Colour : register(b0) { float4 colour; };
[RootSignature(RS)]
float4 main(float4 pos : SV_Position) : SV_Target { return colour; }

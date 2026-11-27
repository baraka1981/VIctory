// victory-metal test vertex shader: float2 positions from vertex buffer
// slot 0, passed through. Pairs with quad_ps.hlsl.
//   dxc -T vs_6_0 -E main quad_vs.hlsl -Fo quad_vs.dxil
#define RS "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), RootConstants(num32BitConstants=4, b0)"
[RootSignature(RS)]
float4 main(float2 pos : POSITION) : SV_Position { return float4(pos, 0, 1); }

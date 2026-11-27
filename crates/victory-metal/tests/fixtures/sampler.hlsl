// victory-metal test kernel: a static sampler (point, clamp) reading a
// 4x1 texture through a descriptor table. out[i] samples u = (i + 0.5) / 4:
// threads 0..3 hit each texel, 4..7 fall past the right edge and clamp to
// the last one (a wrap sampler would repeat instead).
//   dxc -T cs_6_0 -E main sampler.hlsl -Fo sampler.dxil
#define RS "DescriptorTable(SRV(t0), UAV(u0)), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_POINT, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP)"
Texture2D<float4> tex : register(t0);
RWStructuredBuffer<float4> output : register(u0);
SamplerState clamp_point : register(s0);
[RootSignature(RS)]
[numthreads(8, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    output[id.x] = tex.SampleLevel(clamp_point, float2((id.x + 0.5) / 4.0, 0.5), 0);
}

// victory-metal test kernel: resources by index from the directly indexed
// heaps (shader model 6.6), as the translated title shaders do. Root
// constants give the texture's and the output's view heap slots and the
// sampler's sampler heap slot; out[i] samples a 4x1 texture at
// u = (i + 0.5) / 4 with that (point, clamp) sampler.
//   dxc -T cs_6_6 -E main heap.hlsl -Fo heap.dxil
#define RS "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | SAMPLER_HEAP_DIRECTLY_INDEXED), RootConstants(num32BitConstants=3, b0)"
cbuffer Slots : register(b0) { uint texture_slot; uint output_slot; uint sampler_slot; };
[RootSignature(RS)]
[numthreads(8, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    Texture2D<float4> tex = ResourceDescriptorHeap[texture_slot];
    RWStructuredBuffer<float4> output = ResourceDescriptorHeap[output_slot];
    SamplerState point_clamp = SamplerDescriptorHeap[sampler_slot];
    output[id.x] = tex.SampleLevel(point_clamp, float2((id.x + 0.5) / 4.0, 0.5), 0);
}

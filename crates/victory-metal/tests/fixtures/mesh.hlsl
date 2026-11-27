// victory-metal test mesh shader: one group, a quad over the left half
// of the target (two triangles). Pairs with quad_ps.hlsl.
//   dxc -T ms_6_5 -E main mesh.hlsl -Fo mesh.dxil
#define RS "RootConstants(num32BitConstants=4, b0)"
struct Vertex { float4 pos : SV_Position; };
[RootSignature(RS)]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void main(out vertices Vertex verts[4], out indices uint3 tris[2]) {
    SetMeshOutputCounts(4, 2);
    verts[0].pos = float4(-1, -1, 0, 1);
    verts[1].pos = float4(-1, 1, 0, 1);
    verts[2].pos = float4(0, -1, 0, 1);
    verts[3].pos = float4(0, 1, 0, 1);
    tris[0] = uint3(0, 1, 2);
    tris[1] = uint3(2, 1, 3);
}

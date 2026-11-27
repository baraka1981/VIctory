// Root signatures behind the golden bytes in victory-host/src/root.rs tests.
// Built with Microsoft's Linux dxc v1.8.2505:
//   dxc -T rootsig_1_0 -E RS10 root_signatures.hlsl -Fo rs10.bin
//   dxc -T rootsig_1_1 -E RS11 root_signatures.hlsl -Fo rs11.bin
// (the .bin files are not committed; their bytes are pasted into the tests)
#define RS10 "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), RootConstants(num32BitConstants=3, b1, space=2, visibility=SHADER_VISIBILITY_VERTEX), CBV(b0), SRV(t3, space=1), UAV(u0), DescriptorTable(SRV(t0, numDescriptors=4), CBV(b2, offset=4), UAV(u1, numDescriptors=unbounded, offset=8)), DescriptorTable(Sampler(s0, numDescriptors=2), visibility=SHADER_VISIBILITY_PIXEL), StaticSampler(s4, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, comparisonFunc=COMPARISON_LESS, maxAnisotropy=4, mipLODBias=0.5, maxLOD=8)"
#define RS11 "RootConstants(num32BitConstants=1, b0), SRV(t0, flags=DATA_STATIC), UAV(u0, flags=DATA_VOLATILE), DescriptorTable(SRV(t1, numDescriptors=2, flags=DESCRIPTORS_VOLATILE), UAV(u1))"

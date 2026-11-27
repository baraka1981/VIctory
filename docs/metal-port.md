# Metal port notes

Branch `metal-backend` on baraka1981/VIctory. Goal: run on Apple silicon with a Metal backend we own (not D3DMetal).

## Shape

- victory stays a **Windows program running under Wine** (CPU via Rosetta 2, OS via Wine). Loader TLS patch (`gs:[0x58]`) + `clzero` handler should work unchanged since Wine surfaces SIGILL as a Windows exception. Needs macOS 15+ (Rosetta AVX2).
- GPU: `victory-host` = the seam. `victory-gpu` talks to it instead of D3D12 directly.
  - Windows/PC: `victory-gpu/src/backend.rs` (`D3d12`), pass-through.
  - Mac: `victory-metal`, native arm64 lib. Gets called from the Windows side through a Wine unixlib (`__wine_unix_call`), same trick D3DMetal uses. **Not written yet** — see slice 4.
- Shaders: translator already emits DXIL → feed it to Apple's Metal Shader Converter first. Own MSL/AIR emitter off `ir.rs` later.
- Descriptor tables use MSC's `IRDescriptorTableEntry` layout (24 bytes) so MSC-converted shaders read them as is.

## Done

1. Workspace builds + tests on non-Windows (Linux/macOS). Windows unchanged.
2. `victory-host`: `Device` trait — committed / reserved / placed resources, heaps, allocation info, copyable sizes, descriptor heaps + step. `ResourceDesc` moved there, same layout as `D3D12_RESOURCE_DESC` (offsets tested).
3. victory-gpu production creation paths go through it: host textures (clear values, tiled/reserved), depth twins, small-texture heaps, tile heaps, readback/upload/GPU-upload/shared/UAV buffers, RTV/DSV rings, plane pass, view/sampler heaps, null block, magenta fallback.
4. views + samplers in the seam (`shader_view` / `unordered_view` / `constant_view` / `sampler` / `target_view` / `depth_view` / `copy_descriptors` / `descriptor_start`). Descs are D3D12.X's layouts (`ShaderViewDesc` etc., size/align checked against D3D12's). All 23 production view writes in victory-gpu go through it.
5. `victory-metal`: `Device` on Metal (objc2-metal). Format table (DXGI → MTLPixelFormat, numbers cross-checked vs objc2-metal), D3D12-rules copy layouts, private/shared storage by `Memory`, placement heaps, sparse heaps for reserved textures. Views write MSC-style table entries (buffer: addr/0/len, texture: 0/resource id/0, sampler: id/0/lod bias); texture views via `newTextureViewWithPixelFormat:...`, samplers via `MTLSamplerDescriptor` (argument buffer support on); objects a slot needs are kept alive per slot address; RTV/DSV slots remembered for render passes (`MetalDevice::target`). Portable bits unit-tested on Linux.

6. slice 2a: `Recorder` trait (barriers, buffer/resource/texture copies) + `Device::copy_layout` (full footprints). D3D12: `backend::D3d12List`. Wired: submission preludes (`record_transitions`) and depth-twin copies (`record_twin_copy`). Metal: `MetalRecorder` (command buffer + blit encoder; transitions no-op, UAV barrier ends encoder), GPU round-trip test buffer -> texture -> buffer.

7. slice 3a: `ShaderCode` (DXIL / MSL / metallib), `ComputeDesc` (constants at param/index 0, one buffer per binding after), `Device::compute_pipeline`, `Recorder::dispatch`. D3D12: root signature + compute PSO from DXIL. Metal: pipelines from MSL or metallib, compute encoder per dispatch; GPU test runs a real kernel. DXIL on Metal refused until 3b.

8. slice 3b started: `irconverter.rs` loads `libmetalirconverter.dylib` at run time (saxaboom crate; `VICTORY_IRCONVERTER` overrides the path), converts DXIL compute shaders to a metallib (entry renamed `victory_main`, group size from reflection). Converted pipelines (`Abi::Converted`) get a top-level argument buffer at index 2: constants inline, then one 64-bit address per binding; bound buffers made resident with `useResource`. Mac test `converted_dxil_runs_on_the_gpu` (fixture `tests/fixtures/add.dxil`, dxc v1.8.2505 from `add.hlsl`) **passes on the MacBook (27 Nov)**: conversion, reflection group size, and the top-level argument buffer layout (constants inline, then 8-byte-aligned addresses) are confirmed for this model.

## Still direct D3D12 in victory-gpu

- barriers + copies: all production ones go through `Recorder` now (`record_barriers`, `rec(list).copy_*`, with `barrier_of` / `location_of` converting D3D12-built structs). Left direct: diagnostics (`capture.rs`), probes + test helpers (`memory/*`), GPU tests. Next cleanup: build `Barrier`/`Location` at the call sites instead of converting.
- `HostList` itself still holds an `ID3D12GraphicsCommandList`; draws / dispatch / root bindings / clears still direct (2b, after slice 3)
- copy footprints with layouts in `texture.rs`, `texback.rs` (use `copy_layout`)
- title-memory heaps (`inplace.rs`, OpenExistingHeapFromAddress) — Metal analogue: `newBufferWithBytesNoCopy` on page-aligned title memory. Big one for VRAM.
- `CheckFeatureSupport` calls
- diagnostics/probes (`capture.rs`, timestamps, `memory/*_probe.rs`) + GPU tests — fine to stay D3D12

## Next slices

2. ~~2a barriers/copies~~ done. **Order from here: slice 3 before 2b** — Metal can't record a draw or dispatch without a pipeline, so pipelines + root signatures come first.
3. ~~3a compute pipelines + dispatch~~ done. 3b (converter wired + verified on GPU; still to do): turn separately-supplied root signatures (what victory-gpu's translated shaders use) into MSC descriptors, descriptor tables (heap-indexed resources), Metal Shader Converter (`libmetalirconverter.dylib`, Apple developer download) loaded at runtime: DXIL -> metallib, root signature -> top-level argument buffer; then graphics pipelines, and moving victory-gpu's 11 compute PSOs / 14 root signatures onto the seam. Tile mapping for sparse.
4. Wine unixlib bridge: Windows-side proxy impl of `Device` that marshals to `victory-metal`
5. present (CAMetalLayer via winemac.drv's view) + MetalFX instead of FSR

## Checking without the real OS

Rust download server is blocked in the cloud box, so Windows/macOS std comes from source:

```sh
# once: Rust's library/ source for the installed compiler as rust-src
git clone --depth 1 --filter=blob:none --sparse --branch 1.97.0 https://github.com/rust-lang/rust /tmp/rustsrc
(cd /tmp/rustsrc && git sparse-checkout set library)   # + clone backtrace-rs into library/backtrace at the pinned commit
ln -s /tmp/rustsrc/library $(rustc --print sysroot)/lib/rustlib/src/rust/library

# Windows (blake3's C build skipped via its pure mode)
CARGO_FEATURE_PURE=1 RUSTC_BOOTSTRAP=1 cargo check -Zbuild-std=std,panic_abort \
  --target x86_64-pc-windows-msvc --workspace --all-targets [--features victory/diagnostics]

# macOS
RUSTC_BOOTSTRAP=1 cargo check -Zbuild-std=std,panic_abort --target aarch64-apple-darwin -p victory-metal --tests
```

On a real Mac just `cargo test -p victory-metal` (runs the GPU smoke tests in `device.rs`).

Type-checks only. Runtime still needs: Windows + a game for the D3D12 side, a Mac for Metal.

## Gotchas

- `crates/victory-formats/src/bcpack-tables.bin` comes from `Initialize-FormatTables.ps1` (or curl the pinned Marvisak URL in that script, check the SHA-256); bcpack tests fail without it.
- Metal table entry metadata words follow MSC's runtime header from memory — check vs `metal_irconverter_runtime.h` when shaders run. Typed buffer SRV/UAVs need texture-buffer views (TODO), stencil-plane views too.
- Packed 16-bit formats (B5G6R5, BGR5A1, B4G4R4A4) channel order vs Metal unverified — needs a draw-and-read test.
- No D24S8 on Apple silicon → D32FS8 (same as MoltenVK/D3DMetal).
- 16 KB pages under Rosetta/Wine vs the page-guard + DirectStorage tricks: biggest runtime risk, test early.
- Upstream isn't rustfmt-clean; don't mass-format (rebase pain).

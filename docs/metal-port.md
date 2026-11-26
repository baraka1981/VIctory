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
4. `victory-metal`: `Device` on Metal (objc2-metal). Format table (DXGI → MTLPixelFormat, numbers cross-checked vs objc2-metal), D3D12-rules copy layouts, private/shared storage by `Memory`, placement heaps, sparse heaps for reserved textures. Portable bits unit-tested on Linux.

## Still direct D3D12 in victory-gpu

- copy footprints *with layouts* (`texture.rs`, `texback.rs`, `host.rs` `record_twin_copy`) — they feed copy commands, move with slice 2
- title-memory heaps (`inplace.rs`, OpenExistingHeapFromAddress) — Metal analogue: `newBufferWithBytesNoCopy` on page-aligned title memory. Big one for VRAM.
- `CheckFeatureSupport` calls
- views: `CreateShaderResourceView` / UAV / CBV / sampler / RTV / DSV (`descriptor/host.rs`, `host.rs`, `indirect.rs`)
- diagnostics/probes (`capture.rs`, timestamps, `memory/*_probe.rs`) + GPU tests — fine to stay D3D12

## Next slices

1b. views + samplers in the seam (view desc words are D3D12's 24-byte layout already; Metal writes table entries: gpuVA / texture resource id / metadata)
2. command recording: lists, barriers (no-op-ish on Metal), copies, draws/dispatch, footprints
3. pipelines + root signatures (MSC root signature → top-level argument buffer), tile mapping for sparse
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

- `crates/victory-formats/src/bcpack-tables.bin` comes from `Initialize-FormatTables.ps1`; bcpack tests fail without it.
- Packed 16-bit formats (B5G6R5, BGR5A1, B4G4R4A4) channel order vs Metal unverified — needs a draw-and-read test.
- No D24S8 on Apple silicon → D32FS8 (same as MoltenVK/D3DMetal).
- 16 KB pages under Rosetta/Wine vs the page-guard + DirectStorage tricks: biggest runtime risk, test early.
- Upstream isn't rustfmt-clean; don't mass-format (rebase pain).

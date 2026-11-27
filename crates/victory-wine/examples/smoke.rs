//! Slice 4a's smoke test: run under Game Porting Toolkit's Wine with the
//! bridge's folder on `WINEDLLPATH` (`cargo xtask wine` builds it all and
//! prints the command). Runs victory-metal's `add.dxil` (out = in * 2 +
//! base) on Metal through the bridge and checks the result.

#[cfg(windows)]
fn main() {
    if let Err(e) = run() {
        eprintln!("smoke: FAILED: {e}");
        std::process::exit(1);
    }
}

#[cfg(windows)]
fn run() -> Result<(), String> {
    use victory_host::{Memory, ResourceDesc};
    use victory_wine::client::Wine;
    use victory_wine::{Command, Stream, kind};

    const DXIL: &[u8] = include_bytes!("../../victory-metal/tests/fixtures/add.dxil");
    const N: usize = 256;
    const UNORDERED_ACCESS: u32 = 0x4;
    let wine = Wine::load()?;
    println!("smoke: bridge loaded");
    let device = wine.open()?;
    println!("smoke: Metal device open");
    let input = wine.committed(
        device,
        Memory::Upload,
        &ResourceDesc::buffer(N as u64 * 4, 0),
        0,
    )?;
    let output = wine.committed(
        device,
        Memory::Shared,
        &ResourceDesc::buffer(N as u64 * 4, UNORDERED_ACCESS),
        0,
    )?;
    // SAFETY: nothing else holds the input's contents; no GPU work yet.
    let bytes = unsafe { wine.contents(input) }.ok_or("the input buffer has no contents")?;
    for (i, w) in bytes.chunks_exact_mut(4).enumerate() {
        w.copy_from_slice(&(i as u32).to_le_bytes());
    }
    let pipeline = wine.compute_pipeline_dxil(device, DXIL, "main", &[])?;
    println!("smoke: add.dxil converted by Metal Shader Converter");
    let mut s = Stream::default();
    for c in [
        Command::SetComputeRoot(pipeline),
        Command::SetComputePipeline(pipeline),
        Command::SetRootConstants {
            point: 0,
            param: 0,
            offset: 0,
            values: vec![1000],
        },
        Command::SetRootView {
            point: 0,
            param: 1,
            kind: 3,
            address: wine.gpu_address(input),
        },
        Command::SetRootView {
            point: 0,
            param: 2,
            kind: 4,
            address: wine.gpu_address(output),
        },
        Command::Dispatch([(N / 64) as u32, 1, 1]),
    ] {
        c.write(&mut s);
    }
    wine.submit(device, &s, true)?;
    // SAFETY: the GPU is done with it (the submission waited).
    let got = unsafe { wine.contents(output) }.ok_or("the output buffer has no contents")?;
    for (i, w) in got.chunks_exact(4).enumerate() {
        let v = u32::from_le_bytes(w.try_into().unwrap());
        if v != i as u32 * 2 + 1000 {
            return Err(format!("element {i}: {v}, wanted {}", i * 2 + 1000));
        }
    }
    for (k, h) in [
        (kind::PIPELINE, pipeline),
        (kind::RESOURCE, input),
        (kind::RESOURCE, output),
        (kind::DEVICE, device),
    ] {
        wine.release(k, h);
    }
    println!("smoke: OK ({N} elements computed on Metal through Wine)");
    Ok(())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("smoke: a Windows program, run under Wine (see `cargo xtask wine`)");
}

//! `cargo xtask wine`: builds the Wine bridge (slice 4a) into
//! `target/wine/`, laid out as a `WINEDLLPATH` entry:
//!
//! ```text
//! target/wine/x86_64-windows/victory_metal.dll   (empty, marked builtin)
//! target/wine/x86_64-unix/victory_metal.so       (victory-metal, x86_64 macOS)
//! target/wine/smoke.exe                          (the smoke test)
//! ```

use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};

/// What Wine checks for at offset 0x40 (the DOS stub) to call a PE a
/// builtin: then it looks for the `.so` beside it in `<arch>-unix/`.
const BUILTIN: &[u8] = b"Wine builtin DLL\0";

const GPTK_WINE: &str = "/Applications/Game Porting Toolkit.app/Contents/Resources/wine/bin/wine64";

fn cargo(root: &Path, args: &[&str]) -> Result<()> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    println!("wine: cargo {}", args.join(" "));
    let status = Command::new(cargo)
        .args(args)
        .current_dir(root)
        .status()
        .context("running cargo")?;
    ensure!(status.success(), "cargo {} failed", args.join(" "));
    Ok(())
}

/// Writes the builtin marker into a PE's DOS stub.
fn mark_builtin(dll: &mut [u8]) -> Result<()> {
    ensure!(dll.len() > 0x40 && &dll[..2] == b"MZ", "not a PE file");
    let lfanew = u32::from_le_bytes(dll[0x3c..0x40].try_into()?) as usize;
    // The marker must not run into the PE header.
    if lfanew < 0x40 + BUILTIN.len() {
        bail!("the PE header starts at {lfanew:#x}: no room for the builtin marker");
    }
    dll[0x40..0x40 + BUILTIN.len()].copy_from_slice(BUILTIN);
    Ok(())
}

pub fn run(root: &Path, debug: bool) -> Result<()> {
    let profile: &[&str] = if debug { &[] } else { &["--release"] };
    let dir = if debug { "debug" } else { "release" };
    let target = root.join("target");
    let out = target.join("wine");
    fs::create_dir_all(out.join("x86_64-windows"))?;
    fs::create_dir_all(out.join("x86_64-unix"))?;

    let unix = "x86_64-apple-darwin";
    let windows = "x86_64-pc-windows-gnu";
    cargo(
        root,
        &[
            &["build", "-p", "victory-wine-unix", "--target", unix],
            profile,
        ]
        .concat(),
    )?;
    cargo(
        root,
        &[
            &["build", "-p", "victory-wine-dll", "--target", windows],
            profile,
        ]
        .concat(),
    )?;
    cargo(
        root,
        &[
            &[
                "build",
                "-p",
                "victory-wine",
                "--example",
                "smoke",
                "--target",
                windows,
            ],
            profile,
        ]
        .concat(),
    )?;

    let so = target
        .join(unix)
        .join(dir)
        .join("libvictory_wine_unix.dylib");
    fs::copy(&so, out.join("x86_64-unix/victory_metal.so"))
        .with_context(|| format!("copying {}", so.display()))?;
    let dll = target.join(windows).join(dir).join("victory_metal_dll.dll");
    let mut bytes = fs::read(&dll).with_context(|| format!("reading {}", dll.display()))?;
    mark_builtin(&mut bytes)?;
    fs::write(out.join("x86_64-windows/victory_metal.dll"), bytes)?;
    let exe = target.join(windows).join(dir).join("examples/smoke.exe");
    fs::copy(&exe, out.join("smoke.exe")).with_context(|| format!("copying {}", exe.display()))?;

    println!("wine: built into {}", out.display());
    println!("wine: run the smoke test with");
    println!(
        "  WINEDLLPATH=\"{}\" WINEDLLOVERRIDES=victory_metal=b WINEPREFIX=~/.victory-wine \"{GPTK_WINE}\" \"{}\"",
        out.display(),
        out.join("smoke.exe").display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_the_dos_stub() {
        let mut pe = vec![0u8; 0x100];
        pe[..2].copy_from_slice(b"MZ");
        pe[0x3c] = 0x80;
        mark_builtin(&mut pe).unwrap();
        assert_eq!(&pe[0x40..0x51], BUILTIN);
        pe[0x3c] = 0x48;
        assert!(mark_builtin(&mut pe).is_err());
        assert!(mark_builtin(&mut [0u8; 0x100]).is_err());
    }
}

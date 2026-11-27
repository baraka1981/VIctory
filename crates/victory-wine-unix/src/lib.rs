//! `victory_metal.so`: the Unix library of the Wine builtin
//! `victory_metal.dll` (see `victory-wine`). Wine loads it beside the DLL
//! and the Windows side calls its `__wine_unix_call_funcs` entries.
//!
//! Built for x86_64-apple-darwin: Game Porting Toolkit's Wine is an x86_64
//! process under Rosetta, and so is everything loaded into it. Elsewhere
//! this is empty.

#[cfg(target_os = "macos")]
mod bridge;

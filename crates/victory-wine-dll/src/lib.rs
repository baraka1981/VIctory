//! `victory_metal.dll`: nothing in it. It exists so Wine has a builtin
//! module to hang `victory_metal.so` on: `cargo xtask wine` builds it for
//! x86_64-pc-windows-gnu and writes Wine's builtin marker into its DOS
//! stub. The calls themselves are made by `victory-wine`'s client.

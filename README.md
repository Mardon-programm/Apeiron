# Apeiron

A language and cognitive runtime implementation. This repository contains a
workspace of crates covering lexing, parsing, analysis, IR lowering/verification,
execution, effects, builtins, a formatter, a REPL, a language server, project
manifest support, and a CLI.

## Status

- 14 crates, 80 .rs files, ~23k LOC
- 395 tests passing
- `rustfmt --check`, `clippy --all-targets`, and `cargo doc --workspace --no-deps` clean

## Features

- `ffi` (off by default) — resolve native externs declared in `Apeiron.toml`
  under `[native.<name>]` (`library`, `symbol`, `signature`) and bind them
  alongside the standard library when a module runs. Enabled by the CLI so
  `apeiron build` can call into shared libraries via
  [`libloading`](https://crates.io/crates/libloading).
- `faer` (off by default) — back dense matrix multiplication in
  `apeiron-runtime` with the [`faer`](https://crates.io/crates/faer) linear
  algebra library instead of the pure-Rust reference kernel:

  ```bash
  cargo test --workspace --features apeiron-runtime/faer
  ```

## Quick start

```bash
cargo test --workspace
cargo run --quiet --package apeiron-cli -- run path/to/file.ape
cargo run --quiet --package apeiron-cli -- fmt path/to/file.ape
cargo run --quiet --package apeiron-cli -- repl
```

## Structure

- `apeiron-core` — diagnostics, spans, source mapping
- `apeiron-lexer` — tokenization (comments recognized, dropped for tokens but preserved for formatting)
- `apeiron-ast` — AST and language surface
- `apeiron-parser` — concrete syntax → AST
- `apeiron-analyzer` — type/checking, symbols, diagnostics (codes AN0001..AN0018)
- `apeiron-ir` — IR types/instructions/builder/lower/verify/passes
- `apeiron-runtime` — interpreter/execution, memory, causal model, state, native externs (`ffi`)
- `apeiron-effects` — effect analysis of modules
- `apeiron-stdlib` — builtins registry and implementations
- `apeiron-repl` — interactive evaluator
- `apeiron-manifest` — `Apeiron.toml` project loading and linking, `[native.*]` externs
- `apeiron-fmt` — canonical formatter
- `apeiron-lsp` — LSP server over stdio (JSON-RPC framing)
- `apeiron-cli` — binary `apeiron`

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project will be licensed under the same dual license as
above.

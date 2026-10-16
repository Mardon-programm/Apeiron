# Apeiron

A language and cognitive runtime implementation. This repository contains a
workspace of crates covering lexing, parsing, analysis, IR lowering/verification,
execution, effects, builtins, a formatter, a REPL, a language server, project
manifest support, and a CLI.

## Status

- 14 crates, 72 .rs files, ~19k LOC
- 307 tests passing
- `rustfmt --check`, `clippy --all-targets`, and `cargo doc --workspace --no-deps` clean

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
- `apeiron-analyzer` — type/checking, symbols, diagnostics (codes AN0001..AN0016)
- `apeiron-ir` — IR types/instructions/builder/lower/verify/passes
- `apeiron-runtime` — interpreter/execution, memory, causal model, state
- `apeiron-effects` — effect analysis of modules
- `apeiron-stdlib` — builtins registry and implementations
- `apeiron-repl` — interactive evaluator
- `apeiron-manifest` — `Apeiron.toml` project loading and linking
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

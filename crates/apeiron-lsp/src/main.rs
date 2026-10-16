//! The `apeiron-lsp` executable.
//!
//! Runs the language server over stdio. Editors spawn this process and speak
//! JSON-RPC 2.0 with LSP framing; the logic lives in
//! [`apeiron_lsp::run`](apeiron_lsp::run).

use std::io;
use std::process::ExitCode;

fn main() -> ExitCode {
    let stdin = io::stdin();
    let stdout = io::stdout();
    match apeiron_lsp::run(stdin.lock(), stdout.lock()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("apeiron-lsp: {error}");
            ExitCode::FAILURE
        }
    }
}

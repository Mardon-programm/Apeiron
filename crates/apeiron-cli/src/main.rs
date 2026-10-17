//! The `apeiron` executable.
//!
//! Reads a file, dispatches to the pipeline exposed by the library crate, and
//! maps an [`Outcome`](apeiron_cli::Outcome) to an exit code.

use std::fs;
use std::io;
use std::process::ExitCode;

use apeiron_cli::args::{self, Command, USAGE};
use apeiron_cli::{compile, effects, format, parse, tokenize, Options};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let command = match args::parse(&argv) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("error: {message}\n");
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };

    match dispatch(command) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}

fn dispatch(command: Command) -> Result<ExitCode, String> {
    match command {
        Command::Help => {
            print!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        Command::Version => {
            println!("apeiron {VERSION}");
            Ok(ExitCode::SUCCESS)
        }
        Command::Repl => {
            let stdin = io::stdin();
            let mut stdout = io::stdout();
            apeiron_repl::run(stdin.lock(), &mut stdout, true)
                .map_err(|error| format!("repl failed: {error}"))?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Build { path, strict } => {
            let project =
                apeiron_manifest::Project::load(&path).map_err(|error| error.to_string())?;
            let source = project.link().map_err(|error| error.to_string())?;
            let name = format!("{} (linked)", project.name());
            let externs: Vec<_> = project.manifest.externs.values().cloned().collect();
            let outcome = compile(&source, &name, Options { strict, run: true }, &externs);
            emit_diagnostics(&outcome);
            if outcome.has_errors() {
                return Ok(ExitCode::FAILURE);
            }
            let summary = outcome.state_summary();
            if summary.is_empty() {
                println!("built {}: no observable state", project.name());
            } else {
                println!("built {}:", project.name());
                println!("{summary}");
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Tokens { path } => {
            let source = read_file(&path)?;
            match tokenize(&source, &path) {
                Ok(dump) => {
                    print!("{dump}");
                    Ok(ExitCode::SUCCESS)
                }
                Err(rendered) => {
                    eprint!("{rendered}");
                    Ok(ExitCode::FAILURE)
                }
            }
        }
        Command::Parse { path } => {
            let source = read_file(&path)?;
            match parse(&source, &path) {
                Ok(ast) => {
                    print!("{ast}");
                    Ok(ExitCode::SUCCESS)
                }
                Err(rendered) => {
                    eprint!("{rendered}");
                    Ok(ExitCode::FAILURE)
                }
            }
        }
        Command::Check { path, strict } => {
            let source = read_file(&path)?;
            let outcome = compile(&source, &path, Options { strict, run: false }, &[]);
            emit_diagnostics(&outcome);
            if outcome.has_errors() {
                return Ok(ExitCode::FAILURE);
            }
            println!(
                "checked {path}: {} error(s), {} warning(s)",
                outcome.error_count(),
                outcome.warning_count()
            );
            Ok(ExitCode::SUCCESS)
        }
        Command::Ir { path, strict } => {
            let source = read_file(&path)?;
            let outcome = compile(&source, &path, Options { strict, run: false }, &[]);
            emit_diagnostics(&outcome);
            if outcome.has_errors() {
                return Ok(ExitCode::FAILURE);
            }
            if let Some(ir) = &outcome.ir {
                print!("{ir}");
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Effects { path, strict } => {
            let source = read_file(&path)?;
            match effects(&source, &path, strict) {
                Ok(report) => {
                    print!("{report}");
                    Ok(ExitCode::SUCCESS)
                }
                Err(rendered) => {
                    eprint!("{rendered}");
                    Ok(ExitCode::FAILURE)
                }
            }
        }
        Command::Fmt { path, check } => {
            let source = read_file(&path)?;
            let formatted = match format(&source, &path) {
                Ok(formatted) => formatted,
                Err(rendered) => {
                    eprint!("{rendered}");
                    return Ok(ExitCode::FAILURE);
                }
            };
            if check {
                if formatted == source {
                    println!("{path}: already formatted");
                    Ok(ExitCode::SUCCESS)
                } else {
                    eprintln!("{path}: needs formatting");
                    Ok(ExitCode::FAILURE)
                }
            } else if formatted == source {
                println!("{path}: unchanged");
                Ok(ExitCode::SUCCESS)
            } else {
                fs::write(&path, &formatted)
                    .map_err(|error| format!("cannot write `{path}`: {error}"))?;
                println!("{path}: formatted");
                Ok(ExitCode::SUCCESS)
            }
        }
        Command::Run { path, strict } => {
            let source = read_file(&path)?;
            let outcome = compile(&source, &path, Options { strict, run: true }, &[]);
            emit_diagnostics(&outcome);
            if outcome.has_errors() {
                return Ok(ExitCode::FAILURE);
            }
            let summary = outcome.state_summary();
            if summary.is_empty() {
                println!("ran {path}: no observable state");
            } else {
                println!("ran {path}:");
                println!("{summary}");
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn emit_diagnostics(outcome: &apeiron_cli::Outcome) {
    if outcome.diagnostics.is_empty() {
        return;
    }
    eprint!("{}", outcome.render_diagnostics());
}

fn read_file(path: &str) -> Result<String, String> {
    fs::read_to_string(path).map_err(|error| format!("cannot read `{path}`: {error}"))
}

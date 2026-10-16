//! Command-line argument parsing.

/// A parsed command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// Print usage information.
    Help,
    /// Print the version.
    Version,
    /// Compile and execute a file.
    Run {
        /// Input path.
        path: String,
        /// Treat unresolved identifiers as errors.
        strict: bool,
    },
    /// Compile a file without executing it.
    Check {
        /// Input path.
        path: String,
        /// Treat unresolved identifiers as errors.
        strict: bool,
    },
    /// Compile and print the SSA IR.
    Ir {
        /// Input path.
        path: String,
        /// Treat unresolved identifiers as errors.
        strict: bool,
    },
    /// Compile and print the effect summary.
    Effects {
        /// Input path.
        path: String,
        /// Treat unresolved identifiers as errors.
        strict: bool,
    },
    /// Start an interactive session.
    Repl,
    /// Load a project, link it, and execute it.
    Build {
        /// Project directory (defaults to the current directory).
        path: String,
        /// Treat unresolved identifiers as errors.
        strict: bool,
    },
    /// Rewrite a file in canonical style.
    Fmt {
        /// Input path.
        path: String,
        /// Report whether the file is already formatted without rewriting it.
        check: bool,
    },
    /// Parse a file and print its AST.
    Parse {
        /// Input path.
        path: String,
    },
    /// Lex a file and print its tokens.
    Tokens {
        /// Input path.
        path: String,
    },
}

/// Usage text shown by `--help`.
pub const USAGE: &str = "\
apeiron — an AI-native declarative language

USAGE:
    apeiron <COMMAND> <FILE> [OPTIONS]

COMMANDS:
    run      Compile and execute a program
    check    Compile a program and report diagnostics
    ir       Compile a program and print its SSA IR
    effects  Compile a program and print its effect summary
    build    Link and run a project directory
    fmt      Rewrite a program in canonical style
    repl     Start an interactive session
    parse    Parse a program and print its AST
    tokens   Lex a program and print its tokens
    help     Print this help
    version  Print the version

OPTIONS:
    --strict   Treat unresolved identifiers as errors
    --check    With `fmt`, report instead of rewriting
    -h, --help Print help
    -V, --version Print version
";

/// Parses process arguments (excluding the program name).
pub fn parse(args: &[String]) -> Result<Command, String> {
    let mut strict = false;
    let mut check = false;
    let mut positional: Vec<&str> = Vec::new();

    for arg in args {
        match arg.as_str() {
            "-h" | "--help" | "help" => return Ok(Command::Help),
            "-V" | "--version" | "version" => return Ok(Command::Version),
            "--strict" => strict = true,
            "--check" => check = true,
            other if other.starts_with('-') => {
                return Err(format!("unknown option `{other}`"));
            }
            other => positional.push(other),
        }
    }

    if positional.is_empty() {
        return Ok(Command::Help);
    }

    let command = positional[0];
    let path = positional.get(1).map(|value| value.to_string());

    match command {
        "repl" => Ok(Command::Repl),
        "build" => Ok(Command::Build {
            path: path.unwrap_or_else(|| ".".to_string()),
            strict,
        }),
        "run" | "check" | "ir" | "effects" => {
            let path = path.ok_or_else(|| format!("`{command}` requires a file path"))?;
            Ok(match command {
                "run" => Command::Run { path, strict },
                "check" => Command::Check { path, strict },
                "effects" => Command::Effects { path, strict },
                _ => Command::Ir { path, strict },
            })
        }
        "fmt" => {
            let path = path.ok_or_else(|| format!("`{command}` requires a file path"))?;
            Ok(Command::Fmt { path, check })
        }
        "parse" | "tokens" => {
            let path = path.ok_or_else(|| format!("`{command}` requires a file path"))?;
            Ok(if command == "parse" {
                Command::Parse { path }
            } else {
                Command::Tokens { path }
            })
        }
        other => Err(format!("unknown command `{other}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn empty_arguments_show_help() {
        assert_eq!(parse(&[]).unwrap(), Command::Help);
    }

    #[test]
    fn parses_run_with_strict() {
        assert_eq!(
            parse(&args(&["run", "agent.ape", "--strict"])).unwrap(),
            Command::Run {
                path: "agent.ape".into(),
                strict: true,
            }
        );
    }

    #[test]
    fn missing_path_is_an_error() {
        assert!(parse(&args(&["run"])).is_err());
    }

    #[test]
    fn unknown_command_is_an_error() {
        assert!(parse(&args(&["frobnicate", "x.ape"])).is_err());
    }

    #[test]
    fn help_and_version_flags() {
        assert_eq!(parse(&args(&["-h"])).unwrap(), Command::Help);
        assert_eq!(parse(&args(&["--version"])).unwrap(), Command::Version);
    }

    #[test]
    fn parse_command_takes_a_path() {
        assert_eq!(
            parse(&args(&["parse", "x.ape"])).unwrap(),
            Command::Parse {
                path: "x.ape".into()
            }
        );
    }

    #[test]
    fn repl_takes_no_path() {
        assert_eq!(parse(&args(&["repl"])).unwrap(), Command::Repl);
    }

    #[test]
    fn build_defaults_to_the_current_directory() {
        assert_eq!(
            parse(&args(&["build"])).unwrap(),
            Command::Build {
                path: ".".into(),
                strict: false
            }
        );
        assert_eq!(
            parse(&args(&["build", "proj"])).unwrap(),
            Command::Build {
                path: "proj".into(),
                strict: false
            }
        );
    }

    #[test]
    fn fmt_takes_a_path_and_check_flag() {
        assert_eq!(
            parse(&args(&["fmt", "x.ape"])).unwrap(),
            Command::Fmt {
                path: "x.ape".into(),
                check: false
            }
        );
        assert_eq!(
            parse(&args(&["fmt", "x.ape", "--check"])).unwrap(),
            Command::Fmt {
                path: "x.ape".into(),
                check: true
            }
        );
        assert!(parse(&args(&["fmt"])).is_err());
    }
}

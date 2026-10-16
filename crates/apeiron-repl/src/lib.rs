//! An interactive read-eval-print loop for Apeiron.
//!
//! The session is *accumulating*: every accepted line is appended to a growing
//! source buffer and the whole buffer is recompiled and re-run. Declarations
//! therefore persist, and a line that fails to compile is rolled back so the
//! session is never left in a broken state.
//!
//! ```
//! use apeiron_repl::{Reply, Repl};
//!
//! let mut repl = Repl::new();
//! assert_eq!(repl.eval("tensor a : f32[2] = [3.0, 4.0]"), Reply::Quiet);
//! assert_eq!(repl.eval("cause Length -> Threshold"), Reply::Quiet);
//! assert_eq!(repl.eval("observe Length = norm(a)"), Reply::Quiet);
//!
//! match repl.eval(":state") {
//!     Reply::Output(text) => assert!(text.contains("causal")),
//!     other => panic!("expected output, got {other:?}"),
//! }
//! ```

use std::io::{self, BufRead, Write};

use apeiron_analyzer::{analyze_with, AnalysisResult, AnalyzerOptions};
use apeiron_core::diagnostic::{Diagnostic, DiagnosticBag};
use apeiron_core::source::SourceMap;
use apeiron_ir::{lower_program, print_module, Module};
use apeiron_parser::parse_source;
use apeiron_runtime::{run_module_with_builtins, RuntimeState};

/// The documented meta-commands, shown by `:help`.
pub const HELP: &str = "\
:help, :?       show this help
:quit, :q       leave the REPL
:clear          forget everything entered so far
:source         print the accumulated session source
:state          compile, run, and print the resulting state
:ir             compile and print the SSA IR
:effects        compile and print the effect summary";

/// What the REPL did with a single line of input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reply {
    /// The line was consumed and produced no output, as with a declaration.
    Quiet,
    /// Informational text for the caller to print.
    Output(String),
    /// The user asked to end the session.
    Quit,
    /// The line could not be evaluated; the session was left unchanged.
    Error(String),
}

/// An accumulating REPL session.
pub struct Repl {
    name: String,
    source: String,
}

impl Default for Repl {
    fn default() -> Self {
        Self::new()
    }
}

impl Repl {
    /// Starts an empty session.
    pub fn new() -> Self {
        Self {
            name: "<repl>".to_string(),
            source: String::new(),
        }
    }

    /// The source accumulated so far.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Feeds one logical line to the session.
    pub fn eval(&mut self, line: &str) -> Reply {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Reply::Quiet;
        }
        if let Some(command) = trimmed.strip_prefix(':') {
            return self.meta(command.trim());
        }

        let snapshot = self.source.clone();
        self.source.push_str(line);
        if !self.source.ends_with('\n') {
            self.source.push('\n');
        }

        match self.session_state() {
            Ok(_) => Reply::Quiet,
            Err(rendered) => {
                self.source = snapshot;
                Reply::Error(rendered)
            }
        }
    }

    /// Handles a `:command` line.
    fn meta(&mut self, command: &str) -> Reply {
        let name = command.split_whitespace().next().unwrap_or_default();
        match name {
            "help" | "?" => Reply::Output(HELP.to_string()),
            "quit" | "q" | "exit" => Reply::Quit,
            "clear" => {
                self.source.clear();
                Reply::Output("session cleared".to_string())
            }
            "source" => Reply::Output(self.source.clone()),
            "state" => match self.session_state() {
                Ok(state) => Reply::Output(state.summary()),
                Err(rendered) => Reply::Error(rendered),
            },
            "ir" => match self.session_module() {
                Ok(module) => Reply::Output(print_module(&module)),
                Err(rendered) => Reply::Error(rendered),
            },
            "effects" => match self.session_module() {
                Ok(module) => Reply::Output(apeiron_effects::analyze_module(&module).to_string()),
                Err(rendered) => Reply::Error(rendered),
            },
            _ => Reply::Error(format!("unknown command `:{name}`\n{HELP}")),
        }
    }

    /// Compiles the session and returns the lowered module.
    fn session_module(&self) -> Result<Module, String> {
        let mut map = SourceMap::new();
        let file = map.add(&self.name, &self.source);
        let program = parse_source(&self.source, file)
            .map_err(|error| render(Diagnostic::from(error), &map))?;
        let analysis: AnalysisResult = analyze_with(&program, AnalyzerOptions::default());
        if analysis.has_errors() {
            return Err(analysis.render(&map));
        }
        lower_program(&program).map_err(|error| error.to_string())
    }

    /// Compiles and runs the session, yielding its runtime state.
    fn session_state(&self) -> Result<RuntimeState, String> {
        let module = self.session_module()?;
        run_module_with_builtins(&module, apeiron_stdlib::registry())
            .map_err(|error| error.to_string())
    }
}

/// Renders a single diagnostic against a source map.
fn render(diagnostic: Diagnostic, map: &SourceMap) -> String {
    let mut bag = DiagnosticBag::new();
    bag.push(diagnostic);
    bag.render(map)
}

/// Drives a session from `input` to `output`.
///
/// When `interactive`, a banner and prompt are written; otherwise the loop is
/// silent, which is what scripted callers and tests want.
pub fn run<R: BufRead, W: Write>(
    mut input: R,
    output: &mut W,
    interactive: bool,
) -> io::Result<()> {
    let mut repl = Repl::new();
    if interactive {
        writeln!(
            output,
            "Apeiron REPL — type :help for commands, :quit to leave"
        )?;
    }
    let mut line = String::new();
    loop {
        if interactive {
            write!(output, "ape> ")?;
            output.flush()?;
        }
        line.clear();
        if input.read_line(&mut line)? == 0 {
            break;
        }
        let reply = repl.eval(&line);
        match reply {
            Reply::Quiet => {}
            Reply::Output(text) => {
                if !text.is_empty() {
                    writeln!(output, "{text}")?;
                }
            }
            Reply::Error(text) => writeln!(output, "{text}")?,
            Reply::Quit => break,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn declarations_accumulate() {
        let mut repl = Repl::new();
        assert_eq!(repl.eval("tensor a : f32[2] = [3.0, 4.0]"), Reply::Quiet);
        assert_eq!(repl.eval("cause Length -> Threshold"), Reply::Quiet);
        assert_eq!(repl.eval("observe Length = norm(a)"), Reply::Quiet);
        assert!(repl.source().contains("norm(a)"));

        match repl.eval(":state") {
            Reply::Output(text) => assert!(text.contains("causal")),
            other => panic!("expected output, got {other:?}"),
        }
    }

    #[test]
    fn errors_roll_back_the_line() {
        let mut repl = Repl::new();
        assert_eq!(repl.eval("tensor a : f32[2] = [3.0, 4.0]"), Reply::Quiet);
        let before = repl.source().to_string();

        let reply = repl.eval("observe Broken = norm(a");
        assert!(matches!(reply, Reply::Error(_)));
        assert_eq!(repl.source(), before, "the bad line must be rolled back");

        // The prior declaration is still usable.
        assert_eq!(repl.eval("cause Good -> Threshold"), Reply::Quiet);
    }

    #[test]
    fn unknown_meta_command_is_an_error() {
        let mut repl = Repl::new();
        assert!(matches!(repl.eval(":nope"), Reply::Error(_)));
    }

    #[test]
    fn clear_forgets_the_session() {
        let mut repl = Repl::new();
        repl.eval("tensor a : f32[2] = [1.0, 2.0]");
        assert!(matches!(repl.eval(":clear"), Reply::Output(_)));
        assert_eq!(repl.source(), "");
        assert_eq!(repl.eval(":state"), Reply::Output(String::new()));
    }

    #[test]
    fn ir_and_effects_render() {
        let mut repl = Repl::new();
        repl.eval("tensor a : f32[2] = [1.0, 2.0]");
        match repl.eval(":ir") {
            Reply::Output(text) => assert!(text.contains("fn main")),
            other => panic!("expected IR, got {other:?}"),
        }
        match repl.eval(":effects") {
            Reply::Output(text) => assert!(text.contains("module effects")),
            other => panic!("expected effects, got {other:?}"),
        }
    }

    #[test]
    fn driver_reads_until_quit() {
        let input = Cursor::new(
            b"cause Length -> Threshold\nobserve Length = 5.0\n:state\n:quit\nignored\n",
        );
        let mut output = Vec::new();
        run(input, &mut output, false).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("causal"));
        assert!(!text.contains("ignored"));
    }

    #[test]
    fn driver_stops_at_end_of_input() {
        let input = Cursor::new(b"tensor a : f32[2] = [1.0, 2.0]\n");
        let mut output = Vec::new();
        run(input, &mut output, false).unwrap();
        assert!(output.is_empty());
    }
}

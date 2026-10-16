//! The Apeiron command line driver.
//!
//! The crate exposes a small, pure pipeline so it can be unit-tested without
//! touching the filesystem:
//!
//! * [`compile`] runs lex → parse → analyze → lower → verify, and optionally
//!   executes the module, returning an [`Outcome`];
//! * [`tokenize`] and [`parse`] expose the earlier stages for inspection.
//!
//! The binary in `main.rs` is a thin wrapper that reads a file, dispatches a
//! [`Command`](args::Command), and turns an [`Outcome`] into exit codes.

pub mod args;

use apeiron_analyzer::{analyze_with, AnalyzerOptions};
use apeiron_core::diagnostic::{Diagnostic, DiagnosticBag, Severity};
use apeiron_core::source::SourceMap;
use apeiron_effects::analyze_module;
use apeiron_ir::{lower_program, print_module, verify_module};
use apeiron_lexer::Lexer;
use apeiron_parser::parse_source;
use apeiron_runtime::{run_module_with_builtins, RuntimeState};

/// Knobs controlling the pipeline.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Promote unresolved identifiers to errors.
    pub strict: bool,
    /// Execute the module after a successful compile.
    pub run: bool,
}

/// The result of a compiler pipeline invocation.
pub struct Outcome {
    /// Diagnostics gathered across every stage, in stage order.
    pub diagnostics: Vec<Diagnostic>,
    /// The printed IR, when lowering and verification succeeded.
    pub ir: Option<String>,
    /// The final runtime state, when execution ran.
    pub state: Option<RuntimeState>,
    /// Source map used to render diagnostics.
    pub map: SourceMap,
}

impl Outcome {
    /// Whether any error-severity diagnostic was produced.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == Severity::Error)
    }

    /// Number of error diagnostics.
    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == Severity::Error)
            .count()
    }

    /// Number of warning diagnostics.
    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == Severity::Warning)
            .count()
    }

    /// Renders every diagnostic against the source map.
    pub fn render_diagnostics(&self) -> String {
        let mut bag = DiagnosticBag::new();
        bag.extend(self.diagnostics.iter().cloned());
        bag.render(&self.map)
    }

    /// A human readable dump of the runtime state, or an empty string when no
    /// module was executed.
    pub fn state_summary(&self) -> String {
        match &self.state {
            Some(state) => state.summary(),
            None => String::new(),
        }
    }
}

/// Runs the full compiler pipeline (and optionally the runtime).
pub fn compile(source: &str, name: &str, options: Options) -> Outcome {
    let mut map = SourceMap::new();
    let file = map.add(name, source);
    let mut diagnostics = Vec::new();

    let program = match parse_source(source, file) {
        Ok(program) => program,
        Err(error) => {
            diagnostics.push(Diagnostic::from(error));
            return Outcome {
                diagnostics,
                ir: None,
                state: None,
                map,
            };
        }
    };

    let analyzer_options = AnalyzerOptions {
        unresolved_is_error: options.strict,
    };
    let analysis = analyze_with(&program, analyzer_options);
    diagnostics.extend(analysis.diagnostics.iter().cloned());
    if analysis.has_errors() {
        return Outcome {
            diagnostics,
            ir: None,
            state: None,
            map,
        };
    }

    let module = match lower_program(&program) {
        Ok(module) => module,
        Err(error) => {
            diagnostics.push(Diagnostic::error(error.to_string()).with_code("IR0001"));
            return Outcome {
                diagnostics,
                ir: None,
                state: None,
                map,
            };
        }
    };

    if let Err(errors) = verify_module(&module) {
        for error in errors {
            diagnostics.push(Diagnostic::error(error.to_string()).with_code("IR0002"));
        }
        return Outcome {
            diagnostics,
            ir: None,
            state: None,
            map,
        };
    }

    let ir = print_module(&module);

    let state = if options.run {
        match run_module_with_builtins(&module, apeiron_stdlib::registry()) {
            Ok(state) => Some(state),
            Err(error) => {
                diagnostics.push(Diagnostic::error(error.to_string()).with_code("RUN0001"));
                None
            }
        }
    } else {
        None
    };

    Outcome {
        diagnostics,
        ir: Some(ir),
        state,
        map,
    }
}

/// Lexes `source` and returns a textual token dump.
///
/// On failure the returned string is the rendered diagnostics block.
pub fn tokenize(source: &str, name: &str) -> Result<String, String> {
    let mut map = SourceMap::new();
    let file = map.add(name, source);
    let tokens = match Lexer::new(source, file).tokenize() {
        Ok(tokens) => tokens,
        Err(error) => return Err(render_one(Diagnostic::from(error), &map)),
    };
    let mut out = String::new();
    for token in &tokens {
        out.push_str(&format!(
            "{:>5}:{:<4} {}\n",
            token.span.start.line, token.span.start.column, token.kind
        ));
    }
    Ok(out)
}

/// Parses `source` and returns the pretty-printed AST.
///
/// On failure the returned string is the rendered diagnostics block.
pub fn parse(source: &str, name: &str) -> Result<String, String> {
    let mut map = SourceMap::new();
    let file = map.add(name, source);
    match parse_source(source, file) {
        Ok(program) => Ok(program.to_string()),
        Err(error) => Err(render_one(Diagnostic::from(error), &map)),
    }
}

/// Parses, analyzes, and lowers `source`, then summarizes the effects of the
/// resulting module.
///
/// On failure the returned string is the rendered diagnostics block.
pub fn effects(source: &str, name: &str, strict: bool) -> Result<String, String> {
    let mut map = SourceMap::new();
    let file = map.add(name, source);

    let program = match parse_source(source, file) {
        Ok(program) => program,
        Err(error) => return Err(render_one(Diagnostic::from(error), &map)),
    };

    let analysis = analyze_with(
        &program,
        AnalyzerOptions {
            unresolved_is_error: strict,
        },
    );
    if analysis.has_errors() {
        return Err(analysis.render(&map));
    }

    let module = match lower_program(&program) {
        Ok(module) => module,
        Err(error) => return Err(error.to_string()),
    };

    Ok(analyze_module(&module).to_string())
}

/// Rewrites `source` into canonical style.
///
/// On failure the returned string is the rendered diagnostics block.
pub fn format(source: &str, name: &str) -> Result<String, String> {
    let mut map = SourceMap::new();
    let _file = map.add(name, source);
    match apeiron_fmt::format_source(source) {
        Ok(formatted) => Ok(formatted),
        Err(apeiron_fmt::Error::Lex(lex)) => Err(render_one(Diagnostic::from(lex), &map)),
    }
}

fn render_one(diagnostic: Diagnostic, map: &SourceMap) -> String {
    let mut bag = DiagnosticBag::new();
    bag.push(diagnostic);
    bag.render(map)
}

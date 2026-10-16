//! Semantic analysis for Apeiron.
//!
//! The analyzer walks the [`Program`] once, building
//! a lexical [`Scope`] tree, type-checking every expression, and running
//! shape inference over tensor literals and matrix operations. On top of
//! the core type rules it validates the cognitive primitives:
//!
//! * beliefs require a proposition and a confidence, updates require a
//!   likelihood and a reliability;
//! * `observe`/`intervene`/`counterfactual` must reference causal nodes
//!   registered by `cause`;
//! * memory operations must reference declared stores, and clusters may
//!   only contain agents of the same society.
//!
//! All findings are collected as [`Diagnostic`]s with stable `AN...`
//! codes, so the CLI and the LSP can present them identically.

mod check;
mod infer;
pub mod symbol;
pub mod types;

use std::collections::HashSet;

use apeiron_ast::Program;
use apeiron_core::diagnostic::{Diagnostic, DiagnosticBag};
use apeiron_core::source::SourceMap;
use apeiron_core::span::Span;

pub use symbol::{Scope, ScopeKind, Symbol, SymbolKind};
pub use types::{DomainKind, ElemType, TensorType, Type};

/// Knobs controlling how strict the analyzer is.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnalyzerOptions {
    /// Report unresolved identifiers as errors instead of warnings.
    ///
    /// The default is `false` because the domain DSL legitimately
    /// references runtime-provided names such as dataset handles.
    pub unresolved_is_error: bool,
}

/// Incremental semantic analyzer.
pub struct Analyzer {
    pub(crate) scopes: Vec<Scope>,
    pub(crate) diagnostics: DiagnosticBag,
    pub(crate) causal_nodes: HashSet<String>,
    pub(crate) clusters: HashSet<String>,
    pub(crate) options: AnalyzerOptions,
    pub(crate) function_return: Option<Type>,
    pub(crate) loop_depth: usize,
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Analyzer {
    /// Creates an analyzer with default options.
    pub fn new() -> Self {
        Self::with_options(AnalyzerOptions::default())
    }

    /// Creates an analyzer with explicit options.
    pub fn with_options(options: AnalyzerOptions) -> Self {
        Self {
            scopes: Vec::new(),
            diagnostics: DiagnosticBag::new(),
            causal_nodes: HashSet::new(),
            clusters: HashSet::new(),
            options,
            function_return: None,
            loop_depth: 0,
        }
    }

    /// Analyzes a whole program and returns the collected findings.
    pub fn analyze(&mut self, program: &Program) -> AnalysisResult {
        self.push_scope(ScopeKind::Global);
        for item in &program.items {
            self.check_item(item);
        }
        let mut symbols: Vec<Symbol> = self.scopes[0].symbols().cloned().collect();
        symbols.sort_by_key(|symbol| symbol.span.start.offset);
        let diagnostics = std::mem::take(&mut self.diagnostics).into_inner();
        AnalysisResult {
            diagnostics,
            symbols,
            program_span: program.span(),
        }
    }

    // ── scopes ──────────────────────────────────────────────────────

    pub(crate) fn push_scope(&mut self, kind: ScopeKind) {
        self.scopes.push(Scope::new(kind));
    }

    pub(crate) fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub(crate) fn declare(&mut self, symbol: Symbol) {
        let previous = self
            .scopes
            .last_mut()
            .expect("analyzer always has an active scope")
            .insert(symbol.clone());
        if let Some(previous) = previous {
            self.error(
                "AN0002",
                format!(
                    "`{}` is already defined as {} in this scope",
                    symbol.name, previous.kind
                ),
                symbol.span,
            );
        }
    }

    pub(crate) fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.scopes.iter().rev().find_map(|scope| scope.get(name))
    }

    // ── diagnostics ─────────────────────────────────────────────────

    pub(crate) fn error(&mut self, code: &str, message: impl Into<String>, span: Span) {
        self.diagnostics
            .push(Diagnostic::error(message).with_code(code).with_span(span));
    }

    pub(crate) fn warning(&mut self, code: &str, message: impl Into<String>, span: Span) {
        self.diagnostics
            .push(Diagnostic::warning(message).with_code(code).with_span(span));
    }

    pub(crate) fn error_with_help(
        &mut self,
        code: &str,
        message: impl Into<String>,
        span: Span,
        help: impl Into<String>,
    ) {
        self.diagnostics.push(
            Diagnostic::error(message)
                .with_code(code)
                .with_span(span)
                .with_help(help),
        );
    }

    pub(crate) fn unresolved(&mut self, name: &str, span: Span) {
        let message = format!("unresolved identifier `{name}`");
        if self.options.unresolved_is_error {
            self.error("AN0001", message, span);
        } else {
            self.warning("AN0001", message, span);
        }
    }
}

/// Everything the analyzer produced for one program.
#[derive(Clone, Debug)]
pub struct AnalysisResult {
    /// Findings in emission order.
    pub diagnostics: Vec<Diagnostic>,
    /// Top-level (global) symbols in declaration order.
    pub symbols: Vec<Symbol>,
    /// Source range of the analyzed program.
    pub program_span: Span,
}

impl AnalysisResult {
    /// Whether any error-severity diagnostic was reported.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == apeiron_core::diagnostic::Severity::Error)
    }

    /// Whether the program is free of errors (warnings allowed).
    pub fn is_ok(&self) -> bool {
        !self.has_errors()
    }

    /// Number of error diagnostics.
    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == apeiron_core::diagnostic::Severity::Error)
            .count()
    }

    /// Number of warning diagnostics.
    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == apeiron_core::diagnostic::Severity::Warning)
            .count()
    }

    /// Looks up a top-level symbol by name.
    pub fn global(&self, name: &str) -> Option<&Symbol> {
        self.symbols.iter().find(|symbol| symbol.name == name)
    }

    /// Iterates over top-level symbols of a given kind.
    pub fn symbols_of(&self, kind: SymbolKind) -> impl Iterator<Item = &Symbol> {
        self.symbols
            .iter()
            .filter(move |symbol| symbol.kind == kind)
    }

    /// Renders every diagnostic against a source map.
    pub fn render(&self, map: &SourceMap) -> String {
        let mut bag = DiagnosticBag::new();
        bag.extend(self.diagnostics.iter().cloned());
        bag.render(map)
    }
}

/// Analyzes a program with default options.
pub fn analyze(program: &Program) -> AnalysisResult {
    Analyzer::new().analyze(program)
}

/// Analyzes a program with explicit options.
pub fn analyze_with(program: &Program, options: AnalyzerOptions) -> AnalysisResult {
    Analyzer::with_options(options).analyze(program)
}

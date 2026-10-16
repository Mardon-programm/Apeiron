//! Structured diagnostics shared by every compiler stage.
//!
//! Stages build [`Diagnostic`] values instead of formatting strings so
//! that the CLI, the LSP, and tests can all render the same information
//! in the way that suits them.

use std::fmt;
use std::fmt::Write as _;
use std::write as fmt_write;

use crate::source::SourceMap;
use crate::span::Span;

/// How severe a diagnostic is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Severity {
    /// Blocks compilation (`apeiron check`, `apeiron run`).
    Error,
    /// Suspicious but non-blocking finding.
    Warning,
    /// Extra context attached to an existing finding.
    Note,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Error => f.write_str("error"),
            Severity::Warning => f.write_str("warning"),
            Severity::Note => f.write_str("note"),
        }
    }
}

/// A secondary annotated region inside a diagnostic.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Label {
    /// Region of source text being highlighted.
    pub span: Span,
    /// Human explanation printed next to the carets.
    pub message: String,
}

impl Label {
    /// Creates a label for `span`.
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
        }
    }
}

/// One user facing message with an optional source anchor.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Diagnostic {
    /// Severity of this finding.
    pub severity: Severity,
    /// Stable diagnostic code such as `AP0001`, when known.
    pub code: Option<String>,
    /// Main explanation.
    pub message: String,
    /// Primary highlighted span.
    pub span: Option<Span>,
    /// Secondary highlighted regions.
    pub labels: Vec<Label>,
    /// Optional suggestion printed after the source excerpt.
    pub help: Option<String>,
}

impl Diagnostic {
    /// An error diagnostic with only a message.
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(Severity::Error, message)
    }

    /// A warning diagnostic with only a message.
    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, message)
    }

    /// A diagnostic with an explicit severity.
    pub fn new(severity: Severity, message: impl Into<String>) -> Self {
        Self {
            severity,
            code: None,
            message: message.into(),
            span: None,
            labels: Vec::new(),
            help: None,
        }
    }

    /// Assigns a diagnostic code (`AP0001`, `LEX0002`, ...).
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Anchors the diagnostic to a source range.
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Adds a secondary highlighted region.
    pub fn with_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(Label::new(span, message));
        self
    }

    /// Attaches a suggestion or follow-up hint.
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Renders the diagnostic against a source map.
    ///
    /// Falls back to a single header line when the span cannot be
    /// resolved (for example because the file was never registered).
    pub fn render(&self, map: &SourceMap) -> String {
        let mut out = String::new();
        let _ = fmt_write!(out, "{}", self.header());
        if let Some(span) = self.span {
            out.push('\n');
            self.render_excerpt(
                &mut out,
                map,
                span,
                self.labels.first().map(|l| l.message.as_str()),
            );
        }
        for label in &self.labels {
            if Some(label.span) == self.span {
                continue;
            }
            out.push('\n');
            self.render_excerpt(&mut out, map, label.span, Some(label.message.as_str()));
        }
        if let Some(help) = &self.help {
            let _ = fmt_write!(out, "\n  help: {help}");
        }
        out
    }

    /// Single line summary: `error[AP0001]: message`.
    pub fn header(&self) -> String {
        match &self.code {
            Some(code) => format!("{}[{code}]: {}", self.severity, self.message),
            None => format!("{}: {}", self.severity, self.message),
        }
    }

    fn render_excerpt(&self, out: &mut String, map: &SourceMap, span: Span, label: Option<&str>) {
        let Some(file) = map.get(span.file()) else {
            let _ = fmt_write!(out, "  --> <unknown file> {}", span);
            return;
        };
        let line = span.start.line;
        let (start_col, end_col) = if span.end.line == line && span.end.column > span.start.column {
            (span.start.column, span.end.column)
        } else {
            (span.start.column, span.start.column + 1)
        };
        let gutter = line.to_string().len();
        let pad = " ".repeat(gutter);

        let _ = fmt_write!(out, "  --> {}:{}:{}", file.name(), line, span.start.column);
        if let Some(text) = file.line_text(line) {
            let _ = fmt_write!(out, "\n {pad} |");
            let _ = fmt_write!(out, "\n {line} | {text}");
            let indent = " ".repeat((start_col - 1) as usize);
            let width = ((end_col - start_col) as usize).max(1);
            let carets = "^".repeat(width);
            match label {
                Some(msg) if !msg.is_empty() => {
                    let _ = fmt_write!(out, "\n {pad} | {indent}{carets} {msg}");
                }
                _ => {
                    let _ = fmt_write!(out, "\n {pad} | {indent}{carets}");
                }
            }
        }
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.header())
    }
}

/// Ordered collection of diagnostics produced by a compilation run.
#[derive(Clone, Debug, Default)]
pub struct DiagnosticBag {
    diagnostics: Vec<Diagnostic>,
}

impl DiagnosticBag {
    /// Creates an empty bag.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a diagnostic.
    pub fn push(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    /// Appends many diagnostics.
    pub fn extend<I: IntoIterator<Item = Diagnostic>>(&mut self, iter: I) {
        self.diagnostics.extend(iter);
    }

    /// Number of collected diagnostics.
    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    /// Whether nothing was collected.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    /// Whether at least one error-severity diagnostic exists.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
    }

    /// Iterates over diagnostics in emission order.
    pub fn iter(&self) -> std::slice::Iter<'_, Diagnostic> {
        self.diagnostics.iter()
    }

    /// Diagnostics filtered by severity.
    pub fn filter(&self, severity: Severity) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics
            .iter()
            .filter(move |d| d.severity == severity)
    }

    /// Renders every diagnostic separated by blank lines.
    pub fn render(&self, map: &SourceMap) -> String {
        self.diagnostics
            .iter()
            .map(|d| d.render(map))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Consumes the bag, returning all diagnostics.
    pub fn into_inner(self) -> Vec<Diagnostic> {
        self.diagnostics
    }
}

impl From<Diagnostic> for DiagnosticBag {
    fn from(diagnostic: Diagnostic) -> Self {
        let mut bag = DiagnosticBag::new();
        bag.push(diagnostic);
        bag
    }
}

impl IntoIterator for DiagnosticBag {
    type Item = Diagnostic;
    type IntoIter = std::vec::IntoIter<Diagnostic>;

    fn into_iter(self) -> Self::IntoIter {
        self.diagnostics.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{FileId, SourceMap};

    fn anchored_source() -> (SourceMap, Span) {
        let mut map = SourceMap::new();
        let id: FileId = map.add("agent.ape", "let x = )\nlet y = 1");
        let start = map.location(id, 8).unwrap();
        let end = map.location(id, 9).unwrap();
        (map, Span::new(start, end))
    }

    #[test]
    fn header_uses_code_and_severity() {
        let plain = Diagnostic::error("boom");
        assert_eq!(plain.header(), "error: boom");
        let coded = Diagnostic::warning("hm").with_code("AP0002");
        assert_eq!(coded.header(), "warning[AP0002]: hm");
    }

    #[test]
    fn bag_detects_errors() {
        let mut bag = DiagnosticBag::new();
        assert!(!bag.has_errors());
        bag.push(Diagnostic::warning("w"));
        assert!(!bag.has_errors());
        bag.push(Diagnostic::error("e"));
        assert!(bag.has_errors());
        assert_eq!(bag.len(), 2);
        assert_eq!(bag.filter(Severity::Warning).count(), 1);
    }

    #[test]
    fn render_points_at_column() {
        let (map, span) = anchored_source();
        let text = Diagnostic::error("unexpected token")
            .with_code("AP0001")
            .with_span(span)
            .with_label(span, "unexpected token")
            .with_help("remove the stray character")
            .render(&map);
        assert!(text.starts_with("error[AP0001]: unexpected token"));
        assert!(text.contains("--> agent.ape:1:9"));
        assert!(text.contains("| let x = )"));
        assert!(text.contains("^ unexpected token"));
        assert!(text.contains("help: remove the stray character"));
    }

    #[test]
    fn render_without_span_stays_single_line() {
        let text = Diagnostic::error("no anchor").render(&SourceMap::new());
        assert_eq!(text, "error: no anchor");
    }

    #[test]
    fn display_is_header() {
        assert_eq!(Diagnostic::error("m").to_string(), "error: m");
    }
}

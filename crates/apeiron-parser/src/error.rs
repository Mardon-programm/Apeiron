//! Parser error type.

use std::fmt;

use apeiron_core::diagnostic::Diagnostic;
use apeiron_core::span::Span;
use apeiron_lexer::LexError;

/// A syntax error anchored to the offending source range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    /// Explanation shown to the user.
    pub message: String,
    /// Range of source text that could not be parsed.
    pub span: Span,
}

impl ParseError {
    /// Builds an error for `span`.
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (at {})", self.message, self.span)
    }
}

impl std::error::Error for ParseError {}

impl From<LexError> for ParseError {
    fn from(error: LexError) -> Self {
        Self::new(error.message, error.span)
    }
}

impl From<ParseError> for Diagnostic {
    fn from(error: ParseError) -> Self {
        Diagnostic::error(error.message)
            .with_code("PAR0001")
            .with_span(error.span)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apeiron_core::source::FileId;

    #[test]
    fn display_includes_span() {
        let error = ParseError::new("boom", Span::empty(FileId(0)));
        assert_eq!(error.to_string(), "boom (at 0:1:1-1)");
    }

    #[test]
    fn converts_to_diagnostic() {
        let error = ParseError::new("bad", Span::empty(FileId(1)));
        let diag: Diagnostic = error.into();
        assert_eq!(diag.code.as_deref(), Some("PAR0001"));
        assert_eq!(diag.message, "bad");
    }

    #[test]
    fn lex_error_converts() {
        let lex = LexError::new("unterminated", Span::empty(FileId(0)));
        let error: ParseError = lex.into();
        assert_eq!(error.message, "unterminated");
    }
}

//! Error type for failures inside the core layer itself.

use thiserror::Error;

use crate::source::FileId;
use crate::span::Span;

/// Failures produced by [`crate::SourceMap`] operations.
///
/// Higher layers (lexer, parser, analyzer) define their own error types;
/// this one only covers registry lookups and span resolution.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CoreError {
    /// A [`FileId`] was used that was never registered in the map.
    #[error("unknown source file id {0}")]
    UnknownFile(FileId),

    /// A span pointed outside the contents of its file.
    #[error("span {0} lies outside the bounds of its file")]
    SpanOutOfBounds(Span),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_stable() {
        assert_eq!(
            CoreError::UnknownFile(FileId(3)).to_string(),
            "unknown source file id 3"
        );
        let span = Span::empty(FileId(1));
        assert_eq!(
            CoreError::SpanOutOfBounds(span).to_string(),
            "span 1:1:1-1 lies outside the bounds of its file"
        );
    }

    #[test]
    fn is_std_error() {
        fn takes_err(_: &dyn std::error::Error) {}
        takes_err(&CoreError::UnknownFile(FileId(0)));
    }
}

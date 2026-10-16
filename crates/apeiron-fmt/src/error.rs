//! Formatting errors.

use apeiron_lexer::LexError;
use thiserror::Error;

/// Something prevented a source file from being formatted.
#[derive(Debug, Error)]
pub enum Error {
    /// The input could not be tokenized.
    #[error(transparent)]
    Lex(#[from] LexError),
}

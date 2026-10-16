//! Errors raised while loading and linking a project.

use std::io;
use std::path::PathBuf;

/// Something went wrong while reading a manifest or resolving modules.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A manifest file could not be found.
    #[error("no manifest found at `{path}`")]
    NotFound {
        /// The manifest path that was looked for.
        path: PathBuf,
    },
    /// A file could not be read.
    #[error("cannot read `{path}`: {source}")]
    Io {
        /// The offending path.
        path: PathBuf,
        /// The underlying I/O error.
        source: io::Error,
    },
    /// A manifest line did not parse.
    #[error("`{path}:{line}`: {message}")]
    Syntax {
        /// The manifest path.
        path: PathBuf,
        /// One-based line number.
        line: usize,
        /// What was wrong.
        message: String,
    },
    /// A required manifest field was absent.
    #[error("`{path}`: missing required field `{field}`")]
    MissingField {
        /// The manifest path.
        path: PathBuf,
        /// The missing field name.
        field: &'static str,
    },
    /// A key was declared twice.
    #[error("`{path}`: duplicate entry `{key}`")]
    Duplicate {
        /// The manifest path.
        path: PathBuf,
        /// The duplicated key.
        key: String,
    },
    /// Dependencies form a cycle.
    #[error("dependency cycle reached `{path}`")]
    Cycle {
        /// A path already on the dependency stack.
        path: PathBuf,
    },
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, Error>;

//! Runtime errors.

use apeiron_ir::ValueId;

/// What went wrong while executing a module.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RuntimeError {
    /// The module has no `main` function to run.
    #[error("module has no `main` function")]
    NoMain,
    /// A function referenced by a call was not found.
    #[error("unknown function `{name}`")]
    UnknownFunction {
        /// The missing callee.
        name: String,
    },
    /// A block referenced by control flow does not exist.
    #[error("unknown block `{block}`")]
    UnknownBlock {
        /// The missing block index.
        block: u32,
    },
    /// A value was read before being defined.
    #[error("value `{value}` used before it is defined")]
    UndefinedValue {
        /// The offending value.
        value: ValueId,
    },
    /// An operand had the wrong runtime shape or kind.
    #[error("{0}")]
    Type(String),
    /// Arithmetic failed (division by zero, mismatched shapes, ...).
    #[error("{0}")]
    Arithmetic(String),
    /// A domain operation failed (unknown memory store, belief, ...).
    #[error("{0}")]
    Domain(String),
    /// The interpreter stepped on a terminator it cannot continue from.
    #[error("execution reached an unreachable block")]
    Unreachable,
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, RuntimeError>;

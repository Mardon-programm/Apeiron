//! Signatures for Apeiron's built-in standard library.
//!
//! The compiler needs these before any code runs: the analyzer resolves calls
//! against them to type-check arguments, and the lowering stage uses them to
//! give a call its result type. Implementations live in `apeiron-stdlib`;
//! `apeiron-stdlib` verifies at test time that every signature here has one.

/// A primitive result type used by built-in signatures.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BuiltinRet {
    /// No meaningful value.
    Unit,
    /// 64-bit integer.
    I64,
    /// Single precision float.
    F32,
    /// Double precision float.
    F64,
    /// Boolean.
    Bool,
    /// A tensor with an unspecified shape.
    Tensor,
}

/// The signature of a single built-in function.
#[derive(Clone, Copy, Debug)]
pub struct BuiltinSig {
    /// Unique, lower-case name.
    pub name: &'static str,
    /// Expected argument types, in order.
    pub params: &'static [BuiltinRet],
    /// Result type.
    pub ret: BuiltinRet,
}

/// Every built-in Apeiron exposes.
pub const BUILTINS: &[BuiltinSig] = &[
    BuiltinSig {
        name: "cosine_similarity",
        params: &[BuiltinRet::Tensor, BuiltinRet::Tensor],
        ret: BuiltinRet::F32,
    },
    BuiltinSig {
        name: "dot",
        params: &[BuiltinRet::Tensor, BuiltinRet::Tensor],
        ret: BuiltinRet::F32,
    },
    BuiltinSig {
        name: "norm",
        params: &[BuiltinRet::Tensor],
        ret: BuiltinRet::F32,
    },
    BuiltinSig {
        name: "relu",
        params: &[BuiltinRet::Tensor],
        ret: BuiltinRet::Tensor,
    },
    BuiltinSig {
        name: "sigmoid",
        params: &[BuiltinRet::Tensor],
        ret: BuiltinRet::Tensor,
    },
    BuiltinSig {
        name: "tanh",
        params: &[BuiltinRet::Tensor],
        ret: BuiltinRet::Tensor,
    },
    BuiltinSig {
        name: "softmax",
        params: &[BuiltinRet::Tensor],
        ret: BuiltinRet::Tensor,
    },
    BuiltinSig {
        name: "exp",
        params: &[BuiltinRet::Tensor],
        ret: BuiltinRet::Tensor,
    },
    BuiltinSig {
        name: "log",
        params: &[BuiltinRet::Tensor],
        ret: BuiltinRet::Tensor,
    },
    BuiltinSig {
        name: "sqrt",
        params: &[BuiltinRet::Tensor],
        ret: BuiltinRet::Tensor,
    },
    BuiltinSig {
        name: "abs",
        params: &[BuiltinRet::Tensor],
        ret: BuiltinRet::Tensor,
    },
    BuiltinSig {
        name: "transpose",
        params: &[BuiltinRet::Tensor],
        ret: BuiltinRet::Tensor,
    },
    BuiltinSig {
        name: "sum",
        params: &[BuiltinRet::Tensor],
        ret: BuiltinRet::F32,
    },
    BuiltinSig {
        name: "mean",
        params: &[BuiltinRet::Tensor],
        ret: BuiltinRet::F32,
    },
    BuiltinSig {
        name: "argmax",
        params: &[BuiltinRet::Tensor],
        ret: BuiltinRet::I64,
    },
    BuiltinSig {
        name: "zeros",
        params: &[BuiltinRet::I64],
        ret: BuiltinRet::Tensor,
    },
    BuiltinSig {
        name: "ones",
        params: &[BuiltinRet::I64],
        ret: BuiltinRet::Tensor,
    },
];

/// Looks up a built-in by name.
pub fn find(name: &str) -> Option<&'static BuiltinSig> {
    BUILTINS.iter().find(|signature| signature.name == name)
}

/// Whether `name` is a known built-in.
pub fn is_builtin(name: &str) -> bool {
    find(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_known_builtins() {
        assert_eq!(
            find("relu").map(|signature| signature.params.len()),
            Some(1)
        );
        assert_eq!(find("cosine_similarity").unwrap().ret, BuiltinRet::F32);
        assert!(find("nonexistent").is_none());
    }

    #[test]
    fn names_are_unique() {
        for (index, signature) in BUILTINS.iter().enumerate() {
            for other in &BUILTINS[index + 1..] {
                assert_ne!(signature.name, other.name, "duplicate builtin name");
            }
        }
    }

    #[test]
    fn builtin_lookup_agrees_with_is_builtin() {
        for signature in BUILTINS {
            assert!(is_builtin(signature.name));
        }
        assert!(!is_builtin("my_local_fn"));
    }
}

//! Apeiron's native standard library.
//!
//! Built-in functions are part of the language surface rather than runtime
//! artifacts: [`apeiron_ast::builtins`] declares their signatures so that the
//! analyzer and the lowering stage can resolve a call before anything runs,
//! and this crate supplies the implementations.
//!
//! The registry returned by [`registry`] is handed to the evaluator, which
//! falls back to it whenever a call does not resolve to a user-defined
//! function:
//!
//! ```
//! use apeiron_core::source::FileId;
//! use apeiron_ir::lower_program;
//! use apeiron_parser::parse_source;
//! use apeiron_runtime::run_module_with_builtins;
//!
//! let source = r#"
//! tensor a : f32[2] = [3.0, 4.0]
//! cause NormValue -> Threshold
//! observe NormValue = norm(a)
//! "#;
//! let program = parse_source(source, FileId(0)).unwrap();
//! let module = lower_program(&program).unwrap();
//! let state = run_module_with_builtins(&module, apeiron_stdlib::registry()).unwrap();
//! assert_eq!(state.causal.observations()["NormValue"], 5.0);
//! ```

mod ops;

use std::collections::HashMap;

use apeiron_runtime::{NativeFn, Result, Value};

/// The plain function-pointer form of a built-in implementation.
type Impl = fn(&[Value]) -> Result<Value>;

/// Native implementations together with the arity each one enforces.
const REGISTRY: &[(&str, usize, Impl)] = &[
    ("abs", 1, ops::abs),
    ("argmax", 1, ops::argmax),
    ("cosine_similarity", 2, ops::cosine_similarity),
    ("dot", 2, ops::dot),
    ("exp", 1, ops::exp),
    ("log", 1, ops::log),
    ("mean", 1, ops::mean),
    ("norm", 1, ops::norm),
    ("ones", 1, ops::ones),
    ("relu", 1, ops::relu),
    ("sigmoid", 1, ops::sigmoid),
    ("softmax", 1, ops::softmax),
    ("sqrt", 1, ops::sqrt),
    ("sum", 1, ops::sum),
    ("tanh", 1, ops::tanh),
    ("transpose", 1, ops::transpose),
    ("zeros", 1, ops::zeros),
];

/// Builds the registry an evaluator needs to run built-ins.
///
/// A fresh [`HashMap`] per call keeps the registry free of shared state, so
/// callers may mutate it freely.
pub fn registry() -> HashMap<String, NativeFn> {
    REGISTRY
        .iter()
        .map(|(name, _, function)| ((*name).to_string(), Box::new(*function) as NativeFn))
        .collect()
}

/// The implementation of a single built-in, if one exists.
pub fn lookup(name: &str) -> Option<NativeFn> {
    REGISTRY
        .iter()
        .find(|(entry, _, _)| *entry == name)
        .map(|(_, _, function)| Box::new(*function) as NativeFn)
}

/// The number of arguments a built-in demands, if one exists.
pub fn arity(name: &str) -> Option<usize> {
    REGISTRY
        .iter()
        .find(|(entry, _, _)| *entry == name)
        .map(|(_, count, _)| *count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apeiron_ast::builtins::BUILTINS;
    use apeiron_runtime::{RuntimeError, Value};

    #[test]
    fn registry_covers_every_signature() {
        let registry = registry();
        assert_eq!(registry.len(), BUILTINS.len());
        for signature in BUILTINS {
            assert!(
                registry.contains_key(signature.name),
                "missing implementation for `{}`",
                signature.name
            );
        }
    }

    #[test]
    fn declared_arities_match_the_compiler() {
        for (name, count, _) in REGISTRY {
            let signature = BUILTINS
                .iter()
                .find(|signature| signature.name == *name)
                .unwrap_or_else(|| panic!("`{name}` has no signature"));
            assert_eq!(
                signature.params.len(),
                *count,
                "arity mismatch for `{name}`"
            );
            assert_eq!(arity(name), Some(*count));
        }
        assert!(lookup("unknown_builtin").is_none());
        assert_eq!(arity("unknown_builtin"), None);
    }

    #[test]
    fn implementations_reject_the_wrong_argument_count() {
        for (name, count, function) in REGISTRY {
            let too_many = vec![Value::Unit; count + 1];
            let error = function(&too_many)
                .expect_err("an over-supplied call must fail")
                .to_string();
            assert!(
                error.contains("argument(s)"),
                "`{name}` reported {error} without an arity complaint"
            );
        }
    }

    #[test]
    fn implementations_reject_missing_arguments() {
        for (name, count, function) in REGISTRY {
            if *count == 0 {
                continue;
            }
            let error = function(&[])
                .expect_err("an empty argument list must fail")
                .to_string();
            assert!(
                error.contains("argument(s)"),
                "`{name}` reported {error} without an arity complaint"
            );
        }
    }

    #[test]
    fn accepted_arity_falls_through_to_type_checking() {
        for (name, count, function) in REGISTRY {
            // Unit satisfies every arity check, so only a type complaint can follow.
            let unit_args = vec![Value::Unit; *count];
            let error = function(&unit_args)
                .expect_err("unit arguments are never usable")
                .to_string();
            assert!(
                !error.contains("argument(s)"),
                "`{name}` rejected the correct arity: {error}"
            );
        }
    }

    #[test]
    fn lookup_returns_callable_implementations() {
        for (name, _, _) in REGISTRY {
            let function = lookup(name).unwrap_or_else(|| panic!("`{name}` is not registered"));
            let error = function(&[])
                .expect_err("an empty argument list is never valid")
                .to_string();
            assert!(!error.is_empty());
        }
        assert!(lookup("abs").is_some());
    }

    #[test]
    fn wrong_argument_types_are_type_errors() {
        let error = ops::norm(&[Value::Int(1)]).expect_err("an integer is not a tensor");
        assert!(matches!(error, RuntimeError::Type(_)));
        assert!(error.to_string().contains("expected tensor, found i64"));
    }
}

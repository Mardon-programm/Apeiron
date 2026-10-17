//! End-to-end tests for loading native externs declared in `Apeiron.toml`.
//!
//! These tests exercise the `ffi` feature by resolving real symbols from
//! `libm`, which is present on Linux. They compile only when the feature is
//! enabled on a platform that ships `libm.so.6`.

#![cfg(all(feature = "ffi", target_os = "linux"))]

use apeiron_core::{ExternDecl, ExternSig};
use apeiron_runtime::{load_externs, Value};

fn decl(name: &str, symbol: &str, signature: &str) -> ExternDecl {
    ExternDecl {
        name: name.to_string(),
        library: "libm.so.6".to_string(),
        symbol: symbol.to_string(),
        sig: ExternSig::parse(signature).unwrap(),
    }
}

#[test]
fn loads_a_unary_libm_symbol() {
    let mut callables = load_externs(&[decl("sqrt", "sqrt", "f64(f64)")]).unwrap();
    let sqrt = callables.remove("sqrt").unwrap();
    assert_eq!(sqrt(&[Value::Float(9.0)]).unwrap(), Value::Float(3.0));
}

#[test]
fn loads_a_binary_libm_symbol() {
    let mut callables = load_externs(&[decl("pow", "pow", "f64(f64, f64)")]).unwrap();
    let pow = callables.remove("pow").unwrap();
    let result = pow(&[Value::Float(2.0), Value::Float(10.0)]).unwrap();
    assert_eq!(result, Value::Float(1024.0));
}

#[test]
fn wrong_argument_count_is_an_error() {
    let mut callables = load_externs(&[decl("sqrt", "sqrt", "f64(f64)")]).unwrap();
    let sqrt = callables.remove("sqrt").unwrap();
    assert!(sqrt(&[]).is_err());
}

#[test]
fn a_missing_symbol_is_reported() {
    let error = load_externs(&[decl("nope", "definitely_not_a_symbol", "f64()")])
        .err()
        .expect("an unknown symbol must fail");
    assert!(error.to_string().contains("symbol"));
}

#[test]
fn a_missing_library_is_reported() {
    let decl = ExternDecl {
        name: "x".into(),
        library: "libdefinitely_missing_xyz.so".into(),
        symbol: "x".into(),
        sig: ExternSig::parse("f64()").unwrap(),
    };
    let error = load_externs(&[decl])
        .err()
        .expect("a missing library must fail");
    assert!(error.to_string().contains("library"));
}

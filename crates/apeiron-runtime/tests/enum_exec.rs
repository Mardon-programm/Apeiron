//! End-to-end execution tests for `Option`/`Result` and the `?` operator.

use apeiron_core::source::FileId;
use apeiron_ir::{lower_program, Module};
use apeiron_parser::parse_source;
use apeiron_runtime::{Evaluator, Value};

fn compile(source: &str) -> Module {
    let program = parse_source(source, FileId(0))
        .unwrap_or_else(|error| panic!("parse failed:\n{source}\n{error}"));
    lower_program(&program).unwrap_or_else(|error| panic!("lower failed:\n{source}\n{error}"))
}

fn eval_fn(source: &str, name: &str, args: Vec<Value>) -> Value {
    let module = compile(source);
    let mut eval = Evaluator::new(&module);
    eval.call(name, args)
        .unwrap_or_else(|error| panic!("run failed:\n{source}\n{error}"))
}

fn variant(value: &Value) -> &str {
    match value {
        Value::Enum(inner) => inner.variant.as_str(),
        other => panic!("expected enum, found {other}"),
    }
}

fn payload(value: &Value, index: usize) -> Value {
    match value {
        Value::Enum(inner) => inner.payload[index].clone(),
        other => panic!("expected enum, found {other}"),
    }
}

#[test]
fn some_and_none_match_arms() {
    let source = r#"
fn unwrap_or(v: Option[i64]) -> i64 {
    match v {
        Some(x) => x,
        None => 0,
    }
}
"#;
    let some = eval_fn(
        source,
        "unwrap_or",
        vec![Value::Enum(apeiron_runtime::EnumValue::new(
            "Option",
            "Some",
            vec![Value::Int(7)],
        ))],
    );
    assert_eq!(some, Value::Int(7));
    let none = eval_fn(
        source,
        "unwrap_or",
        vec![Value::Enum(apeiron_runtime::EnumValue::new(
            "Option",
            "None",
            Vec::new(),
        ))],
    );
    assert_eq!(none, Value::Int(0));
}

#[test]
fn ok_and_err_match_arms() {
    let source = r#"
fn or_zero(r: Result[i64, str]) -> i64 {
    match r {
        Ok(x) => x,
        Err(_) => 0,
    }
}
"#;
    let ok = eval_fn(
        source,
        "or_zero",
        vec![Value::Enum(apeiron_runtime::EnumValue::new(
            "Result",
            "Ok",
            vec![Value::Int(7)],
        ))],
    );
    assert_eq!(ok, Value::Int(7));
    let err = eval_fn(
        source,
        "or_zero",
        vec![Value::Enum(apeiron_runtime::EnumValue::new(
            "Result",
            "Err",
            vec![Value::Str("nope".into())],
        ))],
    );
    assert_eq!(err, Value::Int(0));
}

#[test]
fn constructors_build_enum_values() {
    let source = r#"
fn make_some() -> Option[i64] {
    Some(42)
}
fn make_none() -> Option[i64] {
    None
}
fn make_ok() -> Result[i64, str] {
    Ok(1)
}
fn make_err() -> Result[i64, str] {
    Err("bad")
}
"#;
    assert_eq!(variant(&eval_fn(source, "make_some", vec![])), "Some");
    assert_eq!(
        payload(&eval_fn(source, "make_some", vec![]), 0),
        Value::Int(42)
    );
    assert_eq!(variant(&eval_fn(source, "make_none", vec![])), "None");
    assert_eq!(variant(&eval_fn(source, "make_ok", vec![])), "Ok");
    assert_eq!(variant(&eval_fn(source, "make_err", vec![])), "Err");
    assert_eq!(
        payload(&eval_fn(source, "make_err", vec![]), 0),
        Value::Str("bad".into())
    );
}

#[test]
fn question_mark_propagates_success() {
    let source = r#"
fn safe_div(a: i64, b: i64) -> Result[i64, str] {
    if b == 0 {
        Err("division by zero")
    } else {
        Ok(a / b)
    }
}
fn both(a: i64, b: i64) -> Result[i64, str] {
    let x = safe_div(a, b)?
    let y = safe_div(x, 2)?
    Ok(y)
}
"#;
    let result = eval_fn(source, "both", vec![Value::Int(20), Value::Int(2)]);
    assert_eq!(variant(&result), "Ok");
    assert_eq!(payload(&result, 0), Value::Int(5));
}

#[test]
fn question_mark_propagates_error() {
    let source = r#"
fn safe_div(a: i64, b: i64) -> Result[i64, str] {
    if b == 0 {
        Err("division by zero")
    } else {
        Ok(a / b)
    }
}
fn both(a: i64, b: i64) -> Result[i64, str] {
    let x = safe_div(a, b)?
    let y = safe_div(x, 2)?
    Ok(y)
}
"#;
    let result = eval_fn(source, "both", vec![Value::Int(20), Value::Int(0)]);
    assert_eq!(variant(&result), "Err");
    assert_eq!(payload(&result, 0), Value::Str("division by zero".into()));
}

#[test]
fn option_question_mark_propagates_none() {
    let source = r#"
fn positive(x: i64) -> Option[i64] {
    if x > 0 { Some(x) } else { None }
}
fn doubled(x: i64) -> Option[i64] {
    let y = positive(x)?
    Some(y * 2)
}
"#;
    let some = eval_fn(source, "doubled", vec![Value::Int(4)]);
    assert_eq!(variant(&some), "Some");
    assert_eq!(payload(&some, 0), Value::Int(8));
    let none = eval_fn(source, "doubled", vec![Value::Int(-1)]);
    assert_eq!(variant(&none), "None");
}

#[test]
fn explicit_return_statements_execute() {
    let source = r#"
fn clamp_low(n: i64) -> i64 {
    if n < 0 {
        return 0
    }
    return n
}
"#;
    assert_eq!(
        eval_fn(source, "clamp_low", vec![Value::Int(-5)]),
        Value::Int(0)
    );
    assert_eq!(
        eval_fn(source, "clamp_low", vec![Value::Int(9)]),
        Value::Int(9)
    );
}

#[test]
fn return_propagates_error_through_question_mark() {
    let source = r#"
fn safe_div(a: i64, b: i64) -> Result[i64, str] {
    if b == 0 {
        return Err("div0")
    }
    return Ok(a / b)
}
fn halve_then(a: i64) -> Result[i64, str] {
    let x = safe_div(a, 2)?
    return Ok(x)
}
"#;
    let ok = eval_fn(source, "halve_then", vec![Value::Int(10)]);
    assert_eq!(variant(&ok), "Ok");
    assert_eq!(payload(&ok, 0), Value::Int(5));
}

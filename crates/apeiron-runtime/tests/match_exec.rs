//! End-to-end execution tests for the `match` expression.

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

#[test]
fn literal_arms_select_the_matching_branch() {
    let source = r#"
fn kind(n: i64) -> i64 {
    match n {
        1 => 10,
        2 => 20,
        _ => 99
    }
}
"#;
    assert_eq!(eval_fn(source, "kind", vec![Value::Int(1)]), Value::Int(10));
    assert_eq!(eval_fn(source, "kind", vec![Value::Int(2)]), Value::Int(20));
    assert_eq!(eval_fn(source, "kind", vec![Value::Int(7)]), Value::Int(99));
}

#[test]
fn bind_pattern_with_guard_selects_on_value() {
    let source = r#"
fn sign(n: i64) -> i64 {
    match n {
        x if x > 0 => 1,
        x if x < 0 => -1,
        _ => 0
    }
}
"#;
    assert_eq!(eval_fn(source, "sign", vec![Value::Int(5)]), Value::Int(1));
    assert_eq!(
        eval_fn(source, "sign", vec![Value::Int(-3)]),
        Value::Int(-1)
    );
    assert_eq!(eval_fn(source, "sign", vec![Value::Int(0)]), Value::Int(0));
}

#[test]
fn bool_match_is_exhaustive_over_two_literals() {
    let source = r#"
fn to_int(flag: bool) -> i64 {
    match flag {
        true => 1,
        false => 0
    }
}
"#;
    assert_eq!(
        eval_fn(source, "to_int", vec![Value::Bool(true)]),
        Value::Int(1)
    );
    assert_eq!(
        eval_fn(source, "to_int", vec![Value::Bool(false)]),
        Value::Int(0)
    );
}

#[test]
fn string_arms_compare_structure() {
    let source = r#"
fn route(name: str) -> i64 {
    match name {
        "query" => 1,
        "command" => 2,
        _ => 0
    }
}
"#;
    assert_eq!(
        eval_fn(source, "route", vec![Value::Str("query".into())]),
        Value::Int(1)
    );
    assert_eq!(
        eval_fn(source, "route", vec![Value::Str("command".into())]),
        Value::Int(2)
    );
    assert_eq!(
        eval_fn(source, "route", vec![Value::Str("chat".into())]),
        Value::Int(0)
    );
}

#[test]
fn tuple_pattern_destructures_a_tensor() {
    let source = r#"
fn sum_pair() -> f32 {
    tensor pair : f32[2] = [1.5, 2.5]
    match pair {
        (a, b) => a + b,
        _ => 0.0
    }
}
"#;
    assert_eq!(eval_fn(source, "sum_pair", vec![]), Value::Float(4.0));
}

#[test]
fn tuple_literal_pattern_matches_element_wise() {
    let source = r#"
fn classify() -> i64 {
    tensor pair : f32[2] = [1.0, 2.0]
    match pair {
        (1.0, 2.0) => 10,
        _ => 0
    }
}
"#;
    assert_eq!(eval_fn(source, "classify", vec![]), Value::Int(10));
}

#[test]
fn nested_match_selects_inner_arm() {
    let source = r#"
fn nested(n: i64) -> i64 {
    match n {
        0 => match n { 0 => 5, _ => 6 },
        _ => 7
    }
}
"#;
    assert_eq!(
        eval_fn(source, "nested", vec![Value::Int(0)]),
        Value::Int(5)
    );
    assert_eq!(
        eval_fn(source, "nested", vec![Value::Int(3)]),
        Value::Int(7)
    );
}

#[test]
fn match_result_participates_in_arithmetic() {
    let source = r#"
fn weighted(n: i64) -> i64 {
    match n { 1 => 10, _ => 20 } + 1
}
"#;
    assert_eq!(
        eval_fn(source, "weighted", vec![Value::Int(1)]),
        Value::Int(11)
    );
    assert_eq!(
        eval_fn(source, "weighted", vec![Value::Int(4)]),
        Value::Int(21)
    );
}

#[test]
fn guard_failure_falls_through_to_next_arm() {
    let source = r#"
fn bucket(n: i64) -> i64 {
    match n {
        x if x > 100 => 2,
        x if x > 10 => 1,
        _ => 0
    }
}
"#;
    assert_eq!(
        eval_fn(source, "bucket", vec![Value::Int(500)]),
        Value::Int(2)
    );
    assert_eq!(
        eval_fn(source, "bucket", vec![Value::Int(50)]),
        Value::Int(1)
    );
    assert_eq!(
        eval_fn(source, "bucket", vec![Value::Int(5)]),
        Value::Int(0)
    );
}

#[test]
fn match_without_a_covering_arm_is_a_runtime_error() {
    let module = compile(
        r#"
fn only_one(n: i64) -> i64 {
    match n {
        1 => 1,
        2 => 2
    }
}
"#,
    );
    let mut eval = Evaluator::new(&module);
    let error = eval
        .call("only_one", vec![Value::Int(9)])
        .expect_err("no arm matches `9`");
    assert!(error.to_string().contains("unreachable"), "{error}");
}

#[test]
fn negative_numeric_pattern_matches() {
    let source = r#"
fn polarity(n: i64) -> i64 {
    match n {
        -1 => 1,
        1 => 2,
        _ => 0
    }
}
"#;
    assert_eq!(
        eval_fn(source, "polarity", vec![Value::Int(-1)]),
        Value::Int(1)
    );
    assert_eq!(
        eval_fn(source, "polarity", vec![Value::Int(1)]),
        Value::Int(2)
    );
}

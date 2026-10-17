//! End-to-end execution tests for tensor arithmetic: broadcasting and
//! element-wise operations against scalar numbers.

use apeiron_core::source::FileId;
use apeiron_ir::lower_program;
use apeiron_parser::parse_source;
use apeiron_runtime::{Evaluator, Value};

fn eval_fn(source: &str, name: &str, args: Vec<Value>) -> Value {
    let program = parse_source(source, FileId(0))
        .unwrap_or_else(|error| panic!("parse failed:\n{source}\n{error}"));
    let module =
        lower_program(&program).unwrap_or_else(|error| panic!("lower failed:\n{source}\n{error}"));
    let mut eval = Evaluator::new(&module);
    eval.call(name, args)
        .unwrap_or_else(|error| panic!("run failed:\n{source}\n{error}"))
}

fn tensor(value: &Value) -> (Vec<usize>, Vec<f64>) {
    match value {
        Value::Tensor(inner) => (inner.shape.clone(), inner.data.clone()),
        other => panic!("expected tensor, found {other}"),
    }
}

#[test]
fn matrix_plus_vector_broadcasts() {
    let source = r#"
fn add(w: f32[2, 3], b: f32[3]) -> f32[2, 3] {
    w + b
}
"#;
    let w = Value::Tensor(apeiron_runtime::Tensor::new(
        apeiron_ir::ElemType::F32,
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let b = Value::Tensor(apeiron_runtime::Tensor::new(
        apeiron_ir::ElemType::F32,
        vec![3],
        vec![10.0, 20.0, 30.0],
    ));
    let result = eval_fn(source, "add", vec![w, b]);
    assert_eq!(
        tensor(&result),
        (vec![2, 3], vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0])
    );
}

#[test]
fn tensor_times_scalar_is_elementwise() {
    let source = r#"
fn scale(v: f32[3], k: f32) -> f32[3] {
    v * k
}
"#;
    let v = Value::Tensor(apeiron_runtime::Tensor::new(
        apeiron_ir::ElemType::F32,
        vec![3],
        vec![1.0, 2.0, 3.0],
    ));
    let result = eval_fn(source, "scale", vec![v, Value::Float(2.5)]);
    assert_eq!(tensor(&result), (vec![3], vec![2.5, 5.0, 7.5]));
}

#[test]
fn scalar_minus_tensor_keeps_operand_order() {
    let source = r#"
fn shift(v: f32[3], k: f32) -> f32[3] {
    k - v
}
"#;
    let v = Value::Tensor(apeiron_runtime::Tensor::new(
        apeiron_ir::ElemType::F32,
        vec![3],
        vec![1.0, 2.0, 3.0],
    ));
    let result = eval_fn(source, "shift", vec![v, Value::Float(10.0)]);
    assert_eq!(tensor(&result), (vec![3], vec![9.0, 8.0, 7.0]));
}

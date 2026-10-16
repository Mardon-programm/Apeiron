//! End-to-end tests: built-ins resolve in the analyzer, lower with a real
//! result type, and execute against the runtime's native registry.

use apeiron_analyzer::{analyze_with, AnalysisResult, AnalyzerOptions};
use apeiron_core::source::FileId;
use apeiron_ir::{lower_program, Module};
use apeiron_parser::parse_source;
use apeiron_runtime::{run_module_with_builtins, RuntimeState};

fn analyze(source: &str) -> AnalysisResult {
    let program = parse_source(source, FileId(0))
        .unwrap_or_else(|error| panic!("parse failed:\n{source}\n{error}"));
    analyze_with(&program, AnalyzerOptions::default())
}

fn state_of(source: &str) -> RuntimeState {
    let program = parse_source(source, FileId(0))
        .unwrap_or_else(|error| panic!("parse failed:\n{source}\n{error}"));
    let module: Module =
        lower_program(&program).unwrap_or_else(|error| panic!("lower failed:\n{source}\n{error}"));
    run_module_with_builtins(&module, apeiron_stdlib::registry())
        .unwrap_or_else(|error| panic!("run failed:\n{source}\n{error}"))
}

const DOT_PROGRAM: &str = r#"
tensor a : f32[3] = [1.0, 2.0, 3.0]
tensor b : f32[3] = [4.0, 5.0, 6.0]
cause DotValue -> Threshold
observe DotValue = dot(a, b)
"#;

#[test]
fn builtins_resolve_without_diagnostics() {
    let result = analyze(DOT_PROGRAM);
    assert_eq!(
        result.diagnostics.len(),
        0,
        "built-ins should not produce diagnostics"
    );
}

#[test]
fn unknown_functions_still_report() {
    let result = analyze("tensor a : f32[2] = [1.0, 2.0]\nlet z = mystery(a)\n");
    assert!(
        result.warning_count() >= 1,
        "an unknown callee should still warn"
    );
}

#[test]
fn builtin_arity_is_checked() {
    let source = r#"
tensor a : f32[2] = [1.0, 2.0]
tensor b : f32[2] = [3.0, 4.0]
let z = sum(a, b)
"#;
    let result = analyze(source);
    assert!(result.has_errors(), "sum/2 should be an arity error");
}

#[test]
fn dot_executes_and_records_state() {
    let state = state_of(DOT_PROGRAM);
    assert_eq!(state.causal.observations()["DotValue"], 32.0);
}

#[test]
fn chained_builtin_results_execute() {
    let source = r#"
tensor a : f32[2] = [3.0, 4.0]
cause NormValue -> Threshold
observe NormValue = norm(a)
"#;
    let state = state_of(source);
    assert_eq!(state.causal.observations()["NormValue"], 5.0);
}

#[test]
fn tensor_returning_builtins_execute() {
    let source = r#"
tensor a : f32[3] = [-1.0, 2.0, -3.0]
cause SumValue -> Threshold
observe SumValue = sum(relu(a))
"#;
    let state = state_of(source);
    assert_eq!(state.causal.observations()["SumValue"], 2.0);
}

#[test]
fn builtins_flow_into_arithmetic() {
    let source = r#"
tensor a : f32[2] = [3.0, 4.0]
cause ScaledNorm -> Threshold
observe ScaledNorm = norm(a) * 2.0
"#;
    let state = state_of(source);
    assert_eq!(state.causal.observations()["ScaledNorm"], 10.0);
}

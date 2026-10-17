//! End-to-end tests for semantic analysis.
//!
//! The reference snippets from the project spec must analyze without
//! errors. Additional cases poke at shape inference, type errors, and the
//! domain-specific checks.

use apeiron_analyzer::{analyze, analyze_with, AnalysisResult, AnalyzerOptions};
use apeiron_core::source::FileId;
use apeiron_parser::parse_source;

fn analyze_ok(source: &str) -> AnalysisResult {
    let program = parse_source(source, FileId(0))
        .unwrap_or_else(|error| panic!("failed to parse:\n{source}\n\nerror: {error}"));
    analyze(&program)
}

fn codes(result: &AnalysisResult) -> Vec<String> {
    result
        .diagnostics
        .iter()
        .filter_map(|diagnostic| diagnostic.code.clone())
        .collect()
}

const SECTION_A: &str = r#"
tensor W : f32[2, 3] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
parameter b : f32[2] = [0.0, 0.0]
tensor x : f32[3] = [1.0, 2.0, 3.0]
tensor y = (W @ x) + b
gradient loss with_respect_to W
optimizer adam { lr: 0.001, beta1: 0.9 }
train model on dataset_ref
"#;

const SECTION_B: &str = r#"
distribution UserIntent {
    query: 0.65,
    command: 0.25,
    chitchat: 0.10
}

belief TargetThreat {
    proposition: "anomaly_detected",
    confidence: 0.42,
    because: "sensor_delta_exceeded"
}

belief TargetThreat update {
    likelihood: 0.88,
    reliability: 0.95
}
"#;

const SECTION_C: &str = r#"
cause Temperature -> Pressure
cause Pressure -> ValveState
observe Temperature = 85.5
intervene ValveState = 1.0
counterfactual ValveState if Temperature == 100.0
simulate horizon = 50
"#;

const SECTION_D: &str = r#"
memory episodic ShortTermMem {
    dimension: 128,
    metric: "cosine",
    capacity: 10000
}

remember ShortTermMem key = vector_query value = "Context payload"
consolidate ShortTermMem -> LongTermMem threshold = 0.8
recall ShortTermMem query = vector_query top_k = 3

plan RouteOptimization { goal: "minimize_cost", horizon: 10 }
"#;

const SECTION_E: &str = r#"
architecture DynamicCore {
    add component "reflection_module",
    expand memory "ShortTermMem" by 2000,
    change routing "primary_llm -> fallback_llm"
}

society MultiAgentSwarm {
    agent Planner { role: "architect" },
    agent Executor { role: "coder" },
    cluster MainCluster = [Planner, Executor]
}

route Request to MainCluster budget { tokens: 4000, latency_ms: 500 }
"#;

#[test]
fn spec_sections_analyze_without_errors() {
    for (name, source) in [
        ("A", SECTION_A),
        ("B", SECTION_B),
        ("C", SECTION_C),
        ("D", SECTION_D),
        ("E", SECTION_E),
    ] {
        let result = analyze_ok(source);
        assert!(
            result.is_ok(),
            "section {name} produced errors: {:?}",
            result.diagnostics
        );
    }
}

#[test]
fn tensor_pipeline_infers_shapes() {
    let result = analyze_ok(
        r#"
tensor W : f32[2, 3] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
tensor x : f32[3] = [1.0, 2.0, 3.0]
tensor y = (W @ x)
"#,
    );
    assert!(result.is_ok(), "{:?}", result.diagnostics);
    let y = result.global("y").expect("y is declared");
    assert_eq!(y.ty.to_string(), "f32[2]");
}

#[test]
fn matmul_shape_mismatch_is_an_error() {
    let result = analyze_ok(
        r#"
tensor A : f32[2, 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
tensor B : f32[4, 2] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
tensor C = A @ B
"#,
    );
    assert!(result.has_errors());
    assert!(codes(&result).contains(&"AN0004".to_string()));
}

#[test]
fn ragged_tensor_literal_is_an_error() {
    let result = analyze_ok("tensor t = [[1.0, 2.0], [3.0]]\n");
    assert!(result.has_errors());
    assert!(codes(&result).contains(&"AN0005".to_string()));
}

#[test]
fn tensor_initializer_shape_mismatch_is_an_error() {
    let result = analyze_ok("tensor t : f32[2] = [[1.0, 2.0]]\n");
    assert!(result.has_errors());
    assert!(codes(&result).contains(&"AN0004".to_string()));
}

#[test]
fn duplicate_declaration_is_an_error() {
    let result = analyze_ok("let x = 1\nlet x = 2\n");
    assert!(result.has_errors());
    assert!(codes(&result).contains(&"AN0002".to_string()));
}

#[test]
fn assigning_to_tensor_is_an_error() {
    let result = analyze_ok("tensor t : f32[2] = [1.0, 2.0]\nt = [3.0, 4.0]\n");
    assert!(result.has_errors());
    assert!(codes(&result).contains(&"AN0009".to_string()));
}

#[test]
fn function_arity_mismatch_is_an_error() {
    let result = analyze_ok(
        r#"
fn scale(x: f32) -> f32 { x * 2.0 }
let z = scale(1.0, 2.0)
"#,
    );
    assert!(result.has_errors());
    assert!(codes(&result).contains(&"AN0007".to_string()));
}

#[test]
fn return_outside_function_is_an_error() {
    let result = analyze_ok("return 1\n");
    assert!(result.has_errors());
    assert!(codes(&result).contains(&"AN0010".to_string()));
}

#[test]
fn break_outside_loop_is_an_error() {
    let result = analyze_ok("break\n");
    assert!(result.has_errors());
    assert!(codes(&result).contains(&"AN0016".to_string()));
}

#[test]
fn updating_unknown_belief_is_an_error() {
    let result = analyze_ok("belief Ghost update { likelihood: 1.0, reliability: 1.0 }\n");
    assert!(result.has_errors());
    assert!(codes(&result).contains(&"AN0003".to_string()));
}

#[test]
fn belief_declaration_missing_confidence_is_an_error() {
    let result = analyze_ok("belief Threat { proposition: \"x\" }\n");
    assert!(result.has_errors());
    assert!(codes(&result).contains(&"AN0008".to_string()));
}

#[test]
fn cluster_member_must_be_an_agent() {
    let result = analyze_ok(
        r#"
society Swarm {
    agent Planner { role: "architect" },
    cluster Main = [Planner, Ghost]
}
"#,
    );
    assert!(result.has_errors());
    assert!(codes(&result).contains(&"AN0014".to_string()));
}

#[test]
fn route_to_declared_cluster_is_accepted() {
    let result = analyze_ok(
        r#"
society Swarm {
    agent Planner { role: "architect" },
    cluster Main = [Planner]
}
route Request to Main budget { tokens: 100, latency_ms: 50 }
"#,
    );
    assert!(result.is_ok(), "{:?}", result.diagnostics);
}

#[test]
fn route_to_unknown_cluster_warns() {
    let result = analyze_ok("route Request to Nowhere budget { tokens: 1, latency_ms: 1 }\n");
    assert!(result.is_ok());
    assert!(codes(&result).contains(&"AN0015".to_string()));
}

#[test]
fn unknown_memory_store_warns_but_does_not_block() {
    let result = analyze_ok("consolidate ShortTermMem -> LongTermMem threshold = 0.8\n");
    assert!(result.is_ok());
    assert!(codes(&result).contains(&"AN0011".to_string()));
}

#[test]
fn unresolved_identifier_is_a_warning_by_default() {
    let result = analyze_ok("let y = zzz\n");
    assert!(result.is_ok());
    assert_eq!(result.warning_count(), 1);
    assert_eq!(result.error_count(), 0);
}

#[test]
fn unresolved_identifier_can_be_promoted_to_error() {
    let program = parse_source("let y = zzz\n", FileId(0)).expect("parses");
    let result = analyze_with(
        &program,
        AnalyzerOptions {
            unresolved_is_error: true,
            ..Default::default()
        },
    );
    assert!(result.has_errors());
    assert!(codes(&result).contains(&"AN0001".to_string()));
}

#[test]
fn global_symbol_table_records_kinds() {
    let result = analyze_ok(
        r#"
tensor W : f32[2, 2] = [[1.0, 2.0], [3.0, 4.0]]
memory episodic ShortTermMem { dimension: 8 }
cause Temperature -> Pressure
"#,
    );
    assert!(result.is_ok(), "{:?}", result.diagnostics);
    assert_eq!(result.global("W").unwrap().ty.to_string(), "f32[2, 2]");
    assert!(result.global("ShortTermMem").is_some());
    assert!(result.global("Temperature").is_some());
    assert!(result.global("Pressure").is_some());
}

#[test]
fn unary_and_comparison_inference() {
    let result = analyze_ok("let ok = 3.0 > 1.0\nlet n = -5\n");
    assert!(result.is_ok(), "{:?}", result.diagnostics);
    assert_eq!(result.global("ok").unwrap().ty.to_string(), "bool");
    assert_eq!(result.global("n").unwrap().ty.to_string(), "i64");
}

#[test]
fn scalar_element_indexing_yields_element_type() {
    let result = analyze_ok(
        r#"
tensor M : f32[2, 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
tensor row = M[0]
tensor cell = M[0][1]
"#,
    );
    assert!(result.is_ok(), "{:?}", result.diagnostics);
    assert_eq!(result.global("row").unwrap().ty.to_string(), "f32[3]");
    assert_eq!(result.global("cell").unwrap().ty.to_string(), "f32");
}

#[test]
fn match_infers_the_unified_arm_type() {
    let result = analyze_ok("let x = match 1 { 0 => 1, _ => 2 }\n");
    assert!(result.is_ok(), "{:?}", result.diagnostics);
    assert_eq!(result.global("x").unwrap().ty.to_string(), "i64");
}

#[test]
fn match_binds_pattern_names_in_the_arm_body() {
    let result = analyze_ok(
        r#"
fn sign(n: i64) -> i64 {
    match n {
        x if x > 0 => 1,
        _ => 0
    }
}
"#,
    );
    assert!(result.is_ok(), "{:?}", result.diagnostics);
}

#[test]
fn match_tuple_pattern_binds_elements() {
    let result = analyze_ok(
        r#"
fn total() -> f32 {
    tensor pair : f32[2] = [1.0, 2.0]
    match pair {
        (a, b) => a + b,
        _ => 0.0
    }
}
"#,
    );
    assert!(result.is_ok(), "{:?}", result.diagnostics);
}

#[test]
fn non_exhaustive_match_is_an_error() {
    let result = analyze_ok("let x = match 1 { 0 => 1, 1 => 2 }\n");
    assert!(!result.is_ok());
    let codes = codes(&result);
    assert!(codes.contains(&"AN0017".to_string()), "{codes:?}");
}

#[test]
fn bool_match_needs_both_literals() {
    let partial = analyze_ok("let x = match true { true => 1 }\n");
    assert!(!partial.is_ok());
    assert!(codes(&partial).contains(&"AN0017".to_string()));

    let complete = analyze_ok("let x = match true { true => 1, false => 0 }\n");
    assert!(complete.is_ok(), "{:?}", complete.diagnostics);
}

#[test]
fn empty_match_is_an_error() {
    let result = analyze_ok("let x = match 1 { }\n");
    assert!(!result.is_ok());
    assert!(codes(&result).contains(&"AN0017".to_string()));
}

#[test]
fn unreachable_match_arm_is_an_error() {
    let result = analyze_ok("let x = match 1 { _ => 0, 2 => 3 }\n");
    assert!(!result.is_ok());
    assert!(codes(&result).contains(&"AN0018".to_string()));
}

#[test]
fn pattern_type_mismatch_is_an_error() {
    let result = analyze_ok("let x = match 1 { \"oops\" => 1, _ => 2 }\n");
    assert!(!result.is_ok());
    assert!(codes(&result).contains(&"AN0018".to_string()));
}

#[test]
fn non_bool_match_guard_is_an_error() {
    let result = analyze_ok("let x = match 1 { n if n => 1, _ => 2 }\n");
    assert!(!result.is_ok());
    assert!(codes(&result).contains(&"AN0003".to_string()));
}

#[test]
fn string_match_analyzes_without_errors() {
    let result = analyze_ok(
        r#"
fn route(name: str) -> i64 {
    match name {
        "query" => 1,
        "command" => 2,
        _ => 0
    }
}
"#,
    );
    assert!(result.is_ok(), "{:?}", result.diagnostics);
}

#[test]
fn option_constructor_and_none_analyze() {
    let result = analyze_ok(
        r#"
fn get() -> Option[i64] {
    if true { Some(3) } else { None }
}
"#,
    );
    assert!(result.is_ok(), "{:?}", result.diagnostics);
}

#[test]
fn result_constructor_and_propagation_analyze() {
    let result = analyze_ok(
        r#"
fn div(a: i64, b: i64) -> Result[i64, str] {
    if b == 0 { Err("div0") } else { Ok(a / b) }
}
fn do_it() -> Result[i64, str] {
    let x = div(6, 2)?
    let y = div(x, 3)?
    Ok(y)
}
"#,
    );
    assert!(result.is_ok(), "{:?}", result.diagnostics);
}

#[test]
fn match_on_option_analyze() {
    let result = analyze_ok(
        r#"
fn unwrap_or(v: Option[i64]) -> i64 {
    match v {
        Some(x) => x,
        None => 0,
    }
}
"#,
    );
    assert!(result.is_ok(), "{:?}", result.diagnostics);
}

#[test]
fn match_on_result_analyze() {
    let result = analyze_ok(
        r#"
fn map_err(r: Result[i64,str]) -> i64 {
    match r {
        Ok(x) => x,
        Err(_) => 0
    }
}
"#,
    );
    assert!(result.is_ok(), "{:?}", result.diagnostics);
}

#[test]
fn question_mark_outside_function_is_an_error() {
    let result = analyze_ok("let x = Some(1)?\n");
    assert!(!result.is_ok());
    assert!(codes(&result).contains(&"AN0003".to_string()));
}

#[test]
fn question_mark_on_non_enum_is_an_error() {
    let result = analyze_ok("fn f(x: i64) -> i64 { x? }\n");
    assert!(!result.is_ok());
    assert!(codes(&result).contains(&"AN0003".to_string()));
}

#[test]
fn question_mark_requires_matching_return_type() {
    let result = analyze_ok(
        r#"
fn div(a: i64, b: i64) -> Result[i64, str] {
    if b == 0 { Err("div0") } else { Ok(a / b) }
}
fn wrong() -> i64 {
    div(1, 2)?
}
"#,
    );
    assert!(!result.is_ok());
    assert!(codes(&result).contains(&"AN0003".to_string()));
}

#[test]
fn non_exhaustive_option_match_is_an_error() {
    let result = analyze_ok(
        r#"
fn only_some(v: Option[i64]) -> i64 {
    match v {
        Some(x) => x
    }
}
"#,
    );
    assert!(!result.is_ok());
    assert!(codes(&result).contains(&"AN0017".to_string()));
}

#[test]
fn non_exhaustive_result_match_is_an_error() {
    let result = analyze_ok(
        r#"
fn only_ok(r: Result[i64, str]) -> i64 {
    match r {
        Ok(x) => x
    }
}
"#,
    );
    assert!(!result.is_ok());
    assert!(codes(&result).contains(&"AN0017".to_string()));
}

#[test]
fn variant_on_wrong_type_is_an_error() {
    let result = analyze_ok(
        r#"
fn bad(v: i64) -> i64 {
    match v {
        Some(x) => x,
        None => 0
    }
}
"#,
    );
    assert!(!result.is_ok());
    assert!(codes(&result).contains(&"AN0018".to_string()));
}

#[test]
fn tail_return_satisfies_declared_return_type() {
    let result = analyze_ok(
        r#"
fn one() -> i64 {
    return 1
}
"#,
    );
    assert!(result.is_ok(), "{:?}", result.diagnostics);
}

#[test]
fn early_return_with_final_return_analyzes() {
    let result = analyze_ok(
        r#"
fn div(a: i64, b: i64) -> Result[i64, str] {
    if b == 0 {
        return Err("div0")
    }
    return Ok(a / b)
}
"#,
    );
    assert!(result.is_ok(), "{:?}", result.diagnostics);
}

#[test]
fn return_of_wrong_type_is_an_error() {
    let result = analyze_ok(
        r#"
fn bad() -> i64 {
    return "text"
}
"#,
    );
    assert!(!result.is_ok());
    assert!(codes(&result).contains(&"AN0003".to_string()));
}

fn analyze_with_externs(source: &str) -> AnalysisResult {
    let program = parse_source(source, FileId(0)).expect("parses");
    analyze_with(
        &program,
        AnalyzerOptions {
            externs: vec![(
                "cosine".to_string(),
                apeiron_core::ExternSig::parse("f64(f64)").unwrap(),
            )],
            ..Default::default()
        },
    )
}

#[test]
fn a_declared_extern_is_callable() {
    let result = analyze_with_externs("let r = cosine(0.0)\n");
    assert!(result.is_ok(), "{:?}", result.diagnostics);
}

#[test]
fn an_undeclared_extern_is_unresolved() {
    let result = analyze_with_externs("let r = sine(0.0)\n");
    assert!(!result.is_ok() || result.warning_count() >= 1);
    assert!(codes(&result).contains(&"AN0001".to_string()));
}

#[test]
fn an_extern_called_with_the_wrong_arity_is_an_error() {
    let result = analyze_with_externs("let r = cosine(0.0, 1.0)\n");
    assert!(codes(&result).contains(&"AN0007".to_string()));
}

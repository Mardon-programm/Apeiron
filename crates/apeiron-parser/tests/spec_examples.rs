//! End-to-end tests parsing the reference snippets from the project spec.
//!
//! Each test feeds a complete program written in the syntax documented in
//! `docs`/the README into [`apeiron_parser::parse_source`] and asserts
//! both that parsing succeeds and that the pretty printer round-trips the
//! domain structure.

use apeiron_ast::Item;
use apeiron_core::diagnostic::Diagnostic;
use apeiron_core::source::{FileId, SourceMap};
use apeiron_parser::parse_source;

fn parse(source: &str) -> apeiron_ast::Program {
    match parse_source(source, FileId(0)) {
        Ok(program) => program,
        Err(error) => panic!("failed to parse:\n{source}\n\nerror: {error}"),
    }
}

#[test]
fn section_a_tensors_and_training() {
    let source = r#"
tensor W : f32[2, 3] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
parameter b : f32[2] = [0.0, 0.0]
tensor x : f32[3] = [1.0, 2.0, 3.0]
tensor y = (W @ x) + b
gradient loss with_respect_to W
optimizer adam { lr: 0.001, beta1: 0.9 }
train model on dataset_ref
"#;
    let program = parse(source);
    assert_eq!(program.items.len(), 7);
    assert!(matches!(program.items[0], Item::Tensor(_)));
    assert!(matches!(program.items[4], Item::Gradient { .. }));
    assert!(matches!(program.items[5], Item::Optimizer { .. }));
    assert!(matches!(program.items[6], Item::Train { .. }));
}

#[test]
fn section_b_epistemic_beliefs() {
    let source = r#"
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
    let program = parse(source);
    assert_eq!(program.items.len(), 3);
    assert!(matches!(program.items[0], Item::Distribution { .. }));

    let declare = match &program.items[1] {
        Item::Belief(stmt) => stmt,
        other => panic!("expected belief, found {other}"),
    };
    assert!(!declare.is_update());
    assert_eq!(declare.name().name, "TargetThreat");
    assert_eq!(declare.fields().len(), 3);

    let update = match &program.items[2] {
        Item::Belief(stmt) => stmt,
        other => panic!("expected belief, found {other}"),
    };
    assert!(update.is_update());
    assert_eq!(update.fields().len(), 2);
}

#[test]
fn section_c_causal_engine() {
    let source = r#"
cause Temperature -> Pressure
cause Pressure -> ValveState
observe Temperature = 85.5
intervene ValveState = 1.0
counterfactual ValveState if Temperature == 100.0
simulate horizon = 50
"#;
    let program = parse(source);
    assert_eq!(program.items.len(), 6);
    assert!(matches!(program.items[0], Item::Cause { .. }));
    assert!(matches!(program.items[2], Item::Observe { .. }));
    assert!(matches!(program.items[3], Item::Intervene { .. }));
    assert!(matches!(program.items[4], Item::Counterfactual { .. }));
    assert!(matches!(program.items[5], Item::Simulate { .. }));
}

#[test]
fn section_d_memory_and_cognition() {
    let source = r#"
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
    let program = parse(source);
    assert_eq!(program.items.len(), 5);
    assert!(matches!(program.items[0], Item::Memory(_)));
    assert!(matches!(program.items[1], Item::Remember { .. }));
    assert!(matches!(program.items[2], Item::Consolidate { .. }));
    assert!(matches!(program.items[3], Item::Recall { .. }));
    assert!(matches!(program.items[4], Item::Plan { .. }));
}

#[test]
fn section_e_self_modification_and_societies() {
    let source = r#"
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
    let program = parse(source);
    assert_eq!(program.items.len(), 3);

    let architecture = match &program.items[0] {
        Item::Architecture(block) => block,
        other => panic!("expected architecture, found {other}"),
    };
    assert_eq!(architecture.name.name, "DynamicCore");
    assert_eq!(architecture.operations.len(), 3);

    let society = match &program.items[1] {
        Item::Society(block) => block,
        other => panic!("expected society, found {other}"),
    };
    assert_eq!(society.members.len(), 3);

    assert!(matches!(program.items[2], Item::Route { .. }));
}

#[test]
fn printer_round_trips_through_reparse() {
    let source = r#"
tensor y = (W @ x) + b
cause Temperature -> Pressure
observe Temperature = 85.5
route Request to MainCluster budget { tokens: 4000, latency_ms: 500 }
"#;
    let first = parse(source);
    let printed = first.to_string();
    let second = parse(&printed);
    assert_eq!(printed, second.to_string());
}

#[test]
fn parse_error_renders_with_source_map() {
    let mut map = SourceMap::new();
    let file = map.add("agent.ape", "let x 1");
    let error = parse_source("let x 1", file).expect_err("must fail");
    let diagnostic: Diagnostic = error.into();
    let rendered = diagnostic.render(&map);
    assert!(rendered.contains("agent.ape:1:7"), "{rendered}");
    assert!(rendered.contains("expected `=`"), "{rendered}");
}

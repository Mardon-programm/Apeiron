//! End-to-end tests: parse, lower, execute, and inspect the cognitive state.

use apeiron_core::source::FileId;
use apeiron_ir::{lower_program, Module};
use apeiron_parser::parse_source;
use apeiron_runtime::{run_module, Evaluator, RuntimeState, Value};

fn compile(source: &str) -> Module {
    let program = parse_source(source, FileId(0))
        .unwrap_or_else(|error| panic!("parse failed:\n{source}\n{error}"));
    lower_program(&program).unwrap_or_else(|error| panic!("lower failed:\n{source}\n{error}"))
}

fn state_of(source: &str) -> RuntimeState {
    let module = compile(source);
    run_module(&module).unwrap_or_else(|error| panic!("run failed:\n{source}\n{error}"))
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
fn section_a_configures_optimizer_and_training() {
    let state = state_of(SECTION_A);
    assert_eq!(state.optimizers.len(), 1);
    assert_eq!(state.optimizers[0].kind, "adam");
    assert!((state.optimizers[0].settings["lr"] - 0.001).abs() < 1e-12);
    assert_eq!(state.training.len(), 1);
}

#[test]
fn section_b_updates_beliefs_and_distributions() {
    let state = state_of(SECTION_B);
    let distribution = state.distributions.get("UserIntent").expect("distribution");
    assert_eq!(distribution.argmax(), Some("query"));
    let belief = state.beliefs.get("TargetThreat").expect("belief");
    assert!(belief.confidence > 0.42, "posterior should rise");
    assert_eq!(belief.updates, 1);
}

#[test]
fn section_c_builds_a_causal_model() {
    let state = state_of(SECTION_C);
    assert_eq!(state.causal.edge_count(), 2);
    assert!(state.causal.is_ancestor("Temperature", "ValveState"));
    assert_eq!(state.causal.observations()["Temperature"], 85.5);
    assert_eq!(state.causal.interventions()["ValveState"], 1.0);
    assert_eq!(state.causal.simulations().len(), 1);
    assert_eq!(state.causal.counterfactuals().len(), 1);
}

#[test]
fn section_d_uses_memory_and_plans() {
    let state = state_of(SECTION_D);
    let short = state.memories.get("ShortTermMem").expect("short store");
    assert_eq!(short.dimension, 128);
    assert_eq!(short.capacity, 10000);
    assert_eq!(short.len(), 1);
    // Consolidation creates and fills the long-term store.
    let long = state.memories.get("LongTermMem").expect("long store");
    assert_eq!(long.len(), 1);
    assert!(state.last_recall.is_some());
    assert!(state.plans.contains_key("RouteOptimization"));
}

#[test]
fn section_e_populates_architecture_and_society() {
    let state = state_of(SECTION_E);
    assert!(state
        .society
        .societies
        .contains(&"MultiAgentSwarm".to_string()));
    assert_eq!(state.society.agents["Planner"], "architect");
    assert_eq!(state.society.agents["Executor"], "coder");
    assert_eq!(
        state.society.clusters["MainCluster"],
        vec!["Planner", "Executor"]
    );
    assert!(!state.architecture.components.is_empty());
    assert!(!state.architecture.memory_expansions.is_empty());
    assert!(!state.architecture.routing.is_empty());
}

#[test]
fn functions_receive_arguments_and_return_values() {
    let module = compile("fn add(a: i64, b: i64) -> i64 { a + b }\n");
    let mut eval = Evaluator::new(&module);
    let sum = eval
        .call("add", vec![Value::Int(2), Value::Int(3)])
        .unwrap();
    assert_eq!(sum, Value::Int(5));
}

#[test]
fn while_loop_threads_carried_values() {
    let module = compile(
        r#"
fn sum(n: i64) -> i64 {
    let total = 0
    let i = 0
    while i < n {
        total = total + i
        i = i + 1
    }
    total
}
"#,
    );
    let mut eval = Evaluator::new(&module);
    let total = eval.call("sum", vec![Value::Int(5)]).unwrap();
    assert_eq!(total, Value::Int(10));
}

#[test]
fn for_loop_iterates_over_a_tensor() {
    let module = compile(
        r#"
fn tally() -> i64 {
    tensor v : f32[3] = [1.0, 2.0, 3.0]
    let count = 0
    for x in v {
        count = count + 1
    }
    count
}
"#,
    );
    let mut eval = Evaluator::new(&module);
    let count = eval.call("tally", vec![]).unwrap();
    assert_eq!(count, Value::Int(3));
}

#[test]
fn if_expression_selects_a_branch() {
    let module = compile(
        r#"
fn pick(flag: i64) -> f32 {
    if flag > 0 { 1.5 } else { 2.5 }
}
"#,
    );
    let mut eval = Evaluator::new(&module);
    assert_eq!(
        eval.call("pick", vec![Value::Int(1)]).unwrap(),
        Value::Float(1.5)
    );
    assert_eq!(
        eval.call("pick", vec![Value::Int(-1)]).unwrap(),
        Value::Float(2.5)
    );
}

#[test]
fn indexing_and_dimensions_work() {
    let module = compile(
        r#"
fn first() -> f32 {
    tensor v : f32[3] = [4.0, 5.0, 6.0]
    v[0]
}
"#,
    );
    let mut eval = Evaluator::new(&module);
    assert_eq!(eval.call("first", vec![]).unwrap(), Value::Float(4.0));
}

#[test]
fn matmul_produces_expected_result() {
    let module = compile(
        r#"
fn project() -> f32[2] {
    tensor w : f32[2, 2] = [[1.0, 2.0], [3.0, 4.0]]
    tensor x : f32[2] = [1.0, 1.0]
    w @ x
}
"#,
    );
    let mut eval = Evaluator::new(&module);
    let result = eval.call("project", vec![]).unwrap();
    assert_eq!(
        result,
        Value::Tensor(apeiron_runtime::Tensor::new(
            apeiron_ir::ElemType::F32,
            vec![2],
            vec![3.0, 7.0]
        ))
    );
}

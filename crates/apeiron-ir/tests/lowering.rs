//! End-to-end lowering tests: parse the reference snippets, lower them to
//! SSA, and verify the result.

use apeiron_core::source::FileId;
use apeiron_ir::{
    lower_program, print_module, verify_module, DeadCodeElimination, Module, Pass, PassManager,
};
use apeiron_parser::parse_source;

fn lower(source: &str) -> Module {
    let program = parse_source(source, FileId(0))
        .unwrap_or_else(|error| panic!("failed to parse:\n{source}\n\nerror: {error}"));
    let module = lower_program(&program)
        .unwrap_or_else(|error| panic!("failed to lower:\n{source}\n\nerror: {error}"));
    if let Err(errors) = verify_module(&module) {
        panic!("verification failed:\n{errors:#?}");
    }
    module
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
fn spec_sections_lower_and_verify() {
    for (name, source) in [
        ("A", SECTION_A),
        ("B", SECTION_B),
        ("C", SECTION_C),
        ("D", SECTION_D),
        ("E", SECTION_E),
    ] {
        let module = lower(source);
        assert!(
            module.function("main").is_some(),
            "section {name} should have a main function"
        );
    }
}

#[test]
fn tensor_pipeline_builds_globals_and_instructions() {
    let module = lower(SECTION_A);
    assert!(module.global("W").is_some());
    assert!(module.global("b").is_some());
    assert!(module.instruction_count() > 0);
    let text = print_module(&module);
    assert!(text.contains("module apeiron"));
    assert!(text.contains("matmul"));
}

#[test]
fn functions_are_lowered_separately() {
    let module = lower(
        r#"
fn scale(x: f32) -> f32 { x * 2.0 }
let y = scale(3.0)
"#,
    );
    let scale = module.function("scale").expect("scale exists");
    assert_eq!(scale.params.len(), 1);
    assert_eq!(scale.return_type.to_string(), "f32");
    let main = module.function("main").expect("main exists");
    assert!(main
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .any(|instruction| matches!(instruction.kind, apeiron_ir::InstrKind::Call { .. })));
}

#[test]
fn while_loop_threads_carried_values() {
    let module = lower(
        r#"
tensor v : f32[3] = [1.0, 2.0, 3.0]
let total = 0.0
let i = 0
while i < 3 {
    total = total + i
    i = i + 1
}
"#,
    );
    let main = module.function("main").expect("main exists");
    let labels: Vec<&str> = main
        .blocks
        .iter()
        .map(|block| block.label.as_str())
        .collect();
    assert!(labels.contains(&"while.header"));
    assert!(labels.contains(&"while.body"));
    assert!(labels.contains(&"while.exit"));
    let header = main
        .blocks
        .iter()
        .find(|block| block.label == "while.header")
        .unwrap();
    assert_eq!(header.params.len(), 2, "i and total are loop-carried");
}

#[test]
fn for_loop_desugars_to_header_body_step() {
    let module = lower(
        r#"
tensor v : f32[3] = [1.0, 2.0, 3.0]
for x in v {
    let y = x
}
"#,
    );
    let main = module.function("main").expect("main exists");
    let labels: Vec<&str> = main
        .blocks
        .iter()
        .map(|block| block.label.as_str())
        .collect();
    assert!(labels.contains(&"for.header"));
    assert!(labels.contains(&"for.body"));
    assert!(labels.contains(&"for.step"));
    assert!(labels.contains(&"for.exit"));
}

#[test]
fn if_expression_produces_merge_block() {
    let module = lower("let x = if 1.0 > 0.0 { 2.0 } else { 3.0 }\n");
    let main = module.function("main").expect("main exists");
    let labels: Vec<&str> = main
        .blocks
        .iter()
        .map(|block| block.label.as_str())
        .collect();
    assert!(labels.contains(&"if.then"));
    assert!(labels.contains(&"if.else"));
    assert!(labels.contains(&"if.merge"));
}

#[test]
fn dead_code_elimination_preserves_effects() {
    let mut module = lower(SECTION_C);
    let before = module.instruction_count();
    DeadCodeElimination.run(&mut module);
    assert!(module.instruction_count() <= before);
    // Causal effects survive elimination.
    assert!(verify_module(&module).is_ok());
    let main = module.function("main").unwrap();
    assert!(main
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .any(|instruction| matches!(instruction.kind, apeiron_ir::InstrKind::Causal(_))));
}

#[test]
fn default_pipeline_keeps_module_valid() {
    let mut module = lower(SECTION_A);
    PassManager::default_pipeline().run(&mut module);
    assert!(verify_module(&module).is_ok());
}

#[test]
fn print_round_trips_all_sections() {
    for source in [SECTION_A, SECTION_B, SECTION_C, SECTION_D, SECTION_E] {
        let module = lower(source);
        let text = print_module(&module);
        assert!(text.contains("module apeiron"));
        assert!(text.contains("fn main()"));
    }
}

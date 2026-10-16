//! Effect-analysis tests over lowered programs.

use apeiron_core::source::FileId;
use apeiron_effects::{analyze_module, Effect, EffectSet};
use apeiron_ir::lower_program;
use apeiron_parser::parse_source;

fn report(source: &str) -> apeiron_effects::EffectReport {
    let program = parse_source(source, FileId(0)).expect("parses");
    let module = lower_program(&program).expect("lowers");
    analyze_module(&module)
}

#[test]
fn pure_arithmetic_has_no_effects() {
    let report = report("fn add(a: i64, b: i64) -> i64 { a + b }\n");
    assert!(report.is_pure());
    assert!(report.function("add").unwrap().is_pure());
}

#[test]
fn memory_operations_report_the_memory_effect() {
    let report = report("memory episodic Mem { dimension: 4 }\nremember Mem key = k value = 1\n");
    assert!(report.module_effects.contains(Effect::Memory));
}

#[test]
fn causal_programs_report_the_causal_effect() {
    let report = report("cause A -> B\nobserve A = 1.0\n");
    assert!(report.module_effects.contains(Effect::Causal));
}

#[test]
fn effects_propagate_through_calls() {
    let report = report(
        r#"
fn recall_it() -> i64 {
    memory episodic M { dimension: 2 }
    recall M query = q top_k = 1
    0
}

fn wrapper() -> i64 {
    recall_it()
}
"#,
    );
    let wrapper = report.function("wrapper").expect("wrapper");
    assert!(wrapper.is_indirect(), "wrapper has no direct effects");
    assert!(wrapper.transitive.contains(Effect::Memory));
}

#[test]
fn combined_effects_are_union() {
    let report = report(
        r#"
belief B { proposition: "p", confidence: 0.5 }
memory episodic Mem { dimension: 2 }
"#,
    );
    let expected: EffectSet = [Effect::Belief, Effect::Memory].into_iter().collect();
    assert_eq!(report.module_effects, expected);
}

#[test]
fn report_display_is_informative() {
    let report = report("cause A -> B\n");
    let text = report.to_string();
    assert!(text.contains("module effects"));
    assert!(text.contains("causal"));
}

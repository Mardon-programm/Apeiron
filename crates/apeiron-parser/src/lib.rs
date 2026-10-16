//! Parser for the Apeiron language.
//!
//! The crate exposes two entry points:
//!
//! * [`parse_source`] — convenience wrapper that lexes then parses a
//!   string, used by the CLI and by tests;
//! * [`Parser`] — the token-driven parser itself, for callers that
//!   already own a token stream (LSP incremental parsing, fuzzing, ...).
//!
//! ```no_run
//! use apeiron_core::source::FileId;
//! use apeiron_parser::parse_source;
//!
//! let program = parse_source("tensor x = [1.0, 2.0]", FileId(0)).expect("valid");
//! assert_eq!(program.items.len(), 1);
//! ```

pub mod error;
pub mod expr;
pub mod parser;

use apeiron_ast::Program;
use apeiron_core::source::FileId;
use apeiron_lexer::Lexer;

pub use error::ParseError;
pub use parser::Parser;

/// Lexes and parses `source` in one step.
pub fn parse_source(source: &str, file: FileId) -> Result<Program, ParseError> {
    let tokens = Lexer::new(source, file).tokenize()?;
    Parser::new(tokens).parse_program()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> Program {
        parse_source(source, FileId(0)).expect("expected successful parse")
    }

    fn parse_err(source: &str) -> ParseError {
        parse_source(source, FileId(0)).expect_err("expected parse failure")
    }

    #[test]
    fn parses_tensor_declaration_with_type() {
        let program = parse("tensor W : f32[2, 3] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]");
        assert_eq!(program.items.len(), 1);
        assert_eq!(
            program.items[0].to_string(),
            "tensor W : f32[2, 3] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]"
        );
    }

    #[test]
    fn parses_tensor_without_type() {
        let program = parse("tensor y = (W @ x) + b");
        assert_eq!(program.items[0].to_string(), "tensor y = W @ x + b");
    }

    #[test]
    fn parses_parameter_and_gradient() {
        let source = "parameter b : f32[2] = [0.0, 0.0]\ngradient loss with_respect_to W";
        let program = parse(source);
        assert_eq!(program.items.len(), 2);
        assert_eq!(
            program.items[0].to_string(),
            "parameter b : f32[2] = [0.0, 0.0]"
        );
        assert_eq!(
            program.items[1].to_string(),
            "gradient loss with_respect_to W"
        );
    }

    #[test]
    fn parses_optimizer_and_train() {
        let source = "optimizer adam { lr: 0.001, beta1: 0.9 }\ntrain model on dataset_ref";
        let program = parse(source);
        assert_eq!(
            program.items[0].to_string(),
            "optimizer adam { lr: 0.001, beta1: 0.9 }"
        );
        assert_eq!(program.items[1].to_string(), "train model on dataset_ref");
    }

    #[test]
    fn parses_distribution_and_beliefs() {
        let source = "\
distribution UserIntent {\n\
    query: 0.65,\n\
    command: 0.25,\n\
    chitchat: 0.10\n\
}\n\
belief TargetThreat {\n\
    proposition: \"anomaly_detected\",\n\
    confidence: 0.42,\n\
    because: \"sensor_delta_exceeded\"\n\
}\n\
belief TargetThreat update {\n\
    likelihood: 0.88,\n\
    reliability: 0.95\n\
}";
        let program = parse(source);
        assert_eq!(program.items.len(), 3);
        assert_eq!(
            program.items[0].to_string(),
            "distribution UserIntent { query: 0.65, command: 0.25, chitchat: 0.1 }"
        );
        assert_eq!(
            program.items[1].to_string(),
            "belief TargetThreat { proposition: \"anomaly_detected\", confidence: 0.42, because: \"sensor_delta_exceeded\" }"
        );
        assert_eq!(
            program.items[2].to_string(),
            "belief TargetThreat update { likelihood: 0.88, reliability: 0.95 }"
        );
    }

    #[test]
    fn parses_causal_engine() {
        let source = "\
cause Temperature -> Pressure\n\
cause Pressure -> ValveState\n\
observe Temperature = 85.5\n\
intervene ValveState = 1.0\n\
counterfactual ValveState if Temperature == 100.0\n\
simulate horizon = 50";
        let program = parse(source);
        assert_eq!(program.items.len(), 6);
        assert_eq!(
            program.items[0].to_string(),
            "cause Temperature -> Pressure"
        );
        assert_eq!(program.items[2].to_string(), "observe Temperature = 85.5");
        assert_eq!(program.items[3].to_string(), "intervene ValveState = 1.0");
        assert_eq!(
            program.items[4].to_string(),
            "counterfactual ValveState if Temperature == 100.0"
        );
        assert_eq!(program.items[5].to_string(), "simulate horizon = 50");
    }

    #[test]
    fn parses_memory_operations() {
        let source = "\
memory episodic ShortTermMem {\n\
    dimension: 128,\n\
    metric: \"cosine\",\n\
    capacity: 10000\n\
}\n\
remember ShortTermMem key = vector_query value = \"Context payload\"\n\
consolidate ShortTermMem -> LongTermMem threshold = 0.8\n\
recall ShortTermMem query = vector_query top_k = 3";
        let program = parse(source);
        assert_eq!(program.items.len(), 4);
        assert_eq!(
            program.items[0].to_string(),
            "memory episodic ShortTermMem { dimension: 128, metric: \"cosine\", capacity: 10000 }"
        );
        assert_eq!(
            program.items[1].to_string(),
            "remember ShortTermMem key = vector_query value = \"Context payload\""
        );
        assert_eq!(
            program.items[2].to_string(),
            "consolidate ShortTermMem -> LongTermMem threshold = 0.8"
        );
        assert_eq!(
            program.items[3].to_string(),
            "recall ShortTermMem query = vector_query top_k = 3"
        );
    }

    #[test]
    fn parses_plan_and_self_model() {
        let source = "plan RouteOptimization { goal: \"minimize_cost\", horizon: 10 }\nself_model Core { identity: \"planner\" }";
        let program = parse(source);
        assert_eq!(
            program.items[0].to_string(),
            "plan RouteOptimization { goal: \"minimize_cost\", horizon: 10 }"
        );
        assert_eq!(
            program.items[1].to_string(),
            "self_model Core { identity: \"planner\" }"
        );
    }

    #[test]
    fn parses_architecture_block() {
        let source = "\
architecture DynamicCore {\n\
    add component \"reflection_module\",\n\
    expand memory \"ShortTermMem\" by 2000,\n\
    change routing \"primary_llm -> fallback_llm\"\n\
}";
        let program = parse(source);
        assert_eq!(
            program.items[0].to_string(),
            "architecture DynamicCore { add component \"reflection_module\", expand memory \"ShortTermMem\" by 2000, change routing \"primary_llm -> fallback_llm\" }"
        );
    }

    #[test]
    fn parses_society_block() {
        let source = "\
society MultiAgentSwarm {\n\
    agent Planner { role: \"architect\" },\n\
    agent Executor { role: \"coder\" },\n\
    cluster MainCluster = [Planner, Executor]\n\
}";
        let program = parse(source);
        assert_eq!(
            program.items[0].to_string(),
            "society MultiAgentSwarm { agent Planner { role: \"architect\" }, agent Executor { role: \"coder\" }, cluster MainCluster = [Planner, Executor] }"
        );
    }

    #[test]
    fn parses_route_with_budget() {
        let program =
            parse("route Request to MainCluster budget { tokens: 4000, latency_ms: 500 }");
        assert_eq!(
            program.items[0].to_string(),
            "route Request to MainCluster budget { tokens: 4000, latency_ms: 500 }"
        );
    }

    #[test]
    fn parses_control_flow() {
        let source = "fn scale(x: f32[3], factor) -> f32[3] { return x @ factor }\nwhile a < b { a = a + 1 }\nfor item in items { process(item) }\nif flag { go() } else if other { stop() } else { wait() }";
        let program = parse(source);
        assert_eq!(program.items.len(), 4);
        assert_eq!(
            program.items[0].to_string(),
            "fn scale(x: f32[3], factor) -> f32[3] { return x @ factor }"
        );
        assert_eq!(program.items[1].to_string(), "while a < b { a = a + 1 }");
        assert_eq!(
            program.items[2].to_string(),
            "for item in items { process(item) }"
        );
        assert_eq!(
            program.items[3].to_string(),
            "if flag { go() } else if other { stop() } else { wait() }"
        );
    }

    #[test]
    fn parses_block_as_expression_statement() {
        let program = parse("{ let a = 1; let b = 2 }");
        assert_eq!(program.items[0].to_string(), "{ let a = 1; let b = 2 }");
    }

    #[test]
    fn precedence_is_respected() {
        assert_eq!(
            parse("tensor t = 1 + 2 * 3").items[0].to_string(),
            "tensor t = 1 + 2 * 3"
        );
        assert_eq!(
            parse("tensor t = (1 + 2) * 3").items[0].to_string(),
            "tensor t = (1 + 2) * 3"
        );
        assert_eq!(
            parse("tensor t = W @ x + b").items[0].to_string(),
            "tensor t = W @ x + b"
        );
        assert_eq!(
            parse("tensor t = -x + y").items[0].to_string(),
            "tensor t = -x + y"
        );
        assert_eq!(
            parse("tensor t = a == b").items[0].to_string(),
            "tensor t = a == b"
        );
    }

    #[test]
    fn indexing_and_field_access() {
        assert_eq!(
            parse("tensor t = W[0]").items[0].to_string(),
            "tensor t = W[0]"
        );
        assert_eq!(
            parse("let a = swarm.Planner.role").items[0].to_string(),
            "let a = swarm.Planner.role"
        );
    }

    #[test]
    fn semicolons_are_optional() {
        let program = parse("let a = 1; let b = 2;");
        assert_eq!(program.items.len(), 2);
    }

    #[test]
    fn errors_report_expected_token() {
        let error = parse_err("let x 1");
        assert!(
            error.message.starts_with("expected `=`"),
            "{}",
            error.message
        );
        assert_eq!(error.span.start.column, 7);
    }

    #[test]
    fn errors_report_missing_closing_brace() {
        let error = parse_err("plan P { goal: 1");
        assert!(error.message.contains("expected `}`"), "{}", error.message);
    }

    #[test]
    fn errors_report_bad_type_dimension() {
        let error = parse_err("tensor t : f32[] = [1.0]");
        assert!(
            error.message.contains("tensor dimension"),
            "{}",
            error.message
        );
    }

    #[test]
    fn errors_report_unexpected_expression() {
        let error = parse_err("tensor t = +");
        assert!(
            error.message.contains("expected an expression"),
            "{}",
            error.message
        );
    }

    #[test]
    fn errors_report_empty_remember() {
        let error = parse_err("remember Mem");
        assert!(error.message.contains("name = value"), "{}", error.message);
    }

    #[test]
    fn errors_report_bad_memory_tier() {
        let error = parse_err("memory deep M { dimension: 1 }");
        assert!(error.message.contains("episodic"), "{}", error.message);
    }

    #[test]
    fn errors_report_bad_arch_op() {
        let error = parse_err("architecture A { remove component \"x\" }");
        assert!(
            error.message.contains("`add`, `expand`, or `change`"),
            "{}",
            error.message
        );
    }

    #[test]
    fn spans_cover_domain_statements() {
        let source = "observe Temperature = 85.5";
        let program = parse(source);
        assert_eq!(program.items[0].span().start.column, 1);
        assert_eq!(program.items[0].span().end.column, 27);
    }
}

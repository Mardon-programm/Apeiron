//! Parsing tests for the postfix `?` propagation operator.

use apeiron_core::source::FileId;
use apeiron_parser::parse_source;

fn parse(source: &str) -> apeiron_ast::Program {
    match parse_source(source, FileId(0)) {
        Ok(program) => program,
        Err(error) => panic!("failed to parse:\n{source}\n\nerror: {error}"),
    }
}

#[test]
fn parses_postfix_question() {
    let program = parse("fn f(x: i64) -> i64 { g(x)? }\n");
    let text = program.items[0].to_string();
    assert!(text.contains("g(x)?"), "{text}");
}

#[test]
fn question_binds_tighter_than_binary_operators() {
    let program = parse("fn f(x: i64) -> i64 { g(x)? + 1 }\n");
    let text = program.items[0].to_string();
    assert!(text.contains("g(x)? + 1"), "{text}");
}

#[test]
fn parses_chained_question() {
    let program = parse("fn f(x: i64) -> i64 { inner(outer(x)?)? }\n");
    let text = program.items[0].to_string();
    assert!(text.contains("inner(outer(x)?)?"), "{text}");
}

#[test]
fn parses_result_and_option_type_annotations() {
    let program = parse(
        r#"
fn parse(text: str) -> Result[i64, str] { Ok(1) }
fn peek(text: str) -> Option[i64] { None }
"#,
    );
    let text = program.to_string();
    assert!(text.contains("Result[i64, str]"), "{text}");
    assert!(text.contains("Option[i64]"), "{text}");
}

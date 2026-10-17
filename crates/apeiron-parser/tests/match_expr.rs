//! Parsing tests for the `match` expression and its arm patterns.

use apeiron_core::source::FileId;
use apeiron_parser::parse_source;

fn parse(source: &str) -> apeiron_ast::Program {
    match parse_source(source, FileId(0)) {
        Ok(program) => program,
        Err(error) => panic!("failed to parse:\n{source}\n\nerror: {error}"),
    }
}

#[test]
fn parses_literal_and_wildcard_arms() {
    let program = parse(r#"let name = match 1 { 0 => "zero", _ => "other" }"#);
    assert_eq!(program.items.len(), 1);
    let text = program.items[0].to_string();
    assert_eq!(text, r#"let name = match 1 { 0 => "zero", _ => "other" }"#);
}

#[test]
fn parses_bind_pattern_with_guard() {
    let program = parse(
        r#"
fn classify(v) {
    match v {
        n if n > 0 => 1,
        _ => 0
    }
}
"#,
    );
    let text = program.items[0].to_string();
    assert!(text.contains("n if n > 0 => 1"), "{text}");
    assert!(text.contains("_ => 0"), "{text}");
}

#[test]
fn parses_tuple_pattern() {
    let program = parse(
        r#"
let pair = match coordinates {
    (a, b) => a,
    _ => 0
}
"#,
    );
    let text = program.items[0].to_string();
    assert!(text.contains("(a, b) => a"), "{text}");
}

#[test]
fn parses_negative_numeric_pattern() {
    let program = parse(r#"let kind = match delta { -1 => "down", 1 => "up", _ => "flat" }"#);
    let text = program.items[0].to_string();
    assert!(
        text.contains("-1 => down") || text.contains("-1 => \"down\""),
        "{text}"
    );
}

#[test]
fn parses_empty_match_and_nested_match() {
    let program = parse(
        r#"
let outer = match a {
    0 => match b {
        1 => 10,
        _ => 11
    },
    _ => 12
}
"#,
    );
    let text = program.items[0].to_string();
    assert!(text.contains("match b { 1 => 10, _ => 11 }"), "{text}");
}

#[test]
fn match_rejects_missing_arrow() {
    let error =
        parse_source("let x = match 1 { 0 1 }", FileId(0)).expect_err("arm without `=>` must fail");
    assert!(error.to_string().contains("expected `=>`"), "{error}");
}

#[test]
fn match_rejects_missing_pattern() {
    let error =
        parse_source("let x = match 1 { => 2 }", FileId(0)).expect_err("missing pattern must fail");
    assert!(error.to_string().contains("expected a pattern"), "{error}");
}

#[test]
fn match_reparses_after_printing() {
    let source = r#"
fn route(request) {
    match request.kind {
        "query" => handle_query(request),
        "command" => handle_command(request),
        other => handle_other(other)
    }
}
"#;
    let first = parse(source);
    let printed = first.to_string();
    let second = parse(&printed);
    assert_eq!(printed, second.to_string());
}

#[test]
fn parses_variant_patterns() {
    let program = parse(
        r#"
let v = match result {
    Some(x) => x,
    None => 0,
    Ok(value) => value,
    Err(message) => 0
}
"#,
    );
    let text = program.items[0].to_string();
    assert!(text.contains("Some(x) => x"), "{text}");
    assert!(text.contains("None => 0"), "{text}");
    assert!(text.contains("Ok(value) => value"), "{text}");
    assert!(text.contains("Err(message) => 0"), "{text}");
}

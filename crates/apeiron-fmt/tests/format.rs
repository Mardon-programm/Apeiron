//! Property-style checks: formatting must never change the token stream.

use apeiron_fmt::format_source;
use apeiron_lexer::{Lexer, Token};

fn tokens(source: &str) -> Vec<Token> {
    let mut toks = Lexer::new(source, apeiron_core::FileId(0))
        .tokenize()
        .expect("source should be lexable");
    toks.pop(); // drop EOF, whose span is a point and carries no text
    toks
}

/// Significant tokens (kind + exact text) must survive formatting unchanged.
fn assert_token_stable(source: &str) {
    let formatted = format_source(source).expect("source should format");
    let before = tokens(source);
    let after = tokens(&formatted);
    assert_eq!(
        before.len(),
        after.len(),
        "token count changed for:\n{source}\n--- produced ---\n{formatted}"
    );
    for (a, b) in before.iter().zip(&after) {
        assert_eq!(a.kind, b.kind, "kind changed for:\n{formatted}");
        assert_eq!(
            &source[a.span.byte_range()],
            &formatted[b.span.byte_range()],
            "text changed for:\n{formatted}"
        );
    }
    // And the result is a fixed point.
    assert_eq!(
        format_source(&formatted).unwrap(),
        formatted,
        "not idempotent"
    );
}

const SAMPLES: &[&str] = &[
    include_str!("data/agent.ape"),
    include_str!("data/messy.ape"),
    "",
    "let x=1",
    "// only a comment\n",
    "belief X {\nproposition:\"p\",\nconfidence:0.5\n}\n",
    "tensor W : f32[2, 3]\nlet y = W@x - 1.0\n",
];

#[test]
fn formatting_is_token_preserving() {
    for sample in SAMPLES {
        assert_token_stable(sample);
    }
}

#[test]
fn nested_block_comments_are_preserved() {
    let source = "/* outer /* inner */ still */\nlet x = 1\n";
    let out = format_source(source).unwrap();
    assert!(out.starts_with("/* outer /* inner */ still */"));
    assert_eq!(out, "/* outer /* inner */ still */\nlet x = 1\n");
}

#[test]
fn formatting_a_formatted_file_is_a_no_op() {
    let formatted = format_source(SAMPLES[0]).unwrap();
    assert_eq!(format_source(&formatted).unwrap(), formatted);
}

#[test]
fn eof_only_source_is_empty() {
    assert_eq!(format_source("").unwrap(), "");
}

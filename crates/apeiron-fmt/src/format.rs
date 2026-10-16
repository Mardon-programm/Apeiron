//! The formatting pass.
//!
//! The formatter is deliberately *lexical*: it re-emits the token stream with a
//! canonical whitespace policy while leaving the original token text untouched.
//! This keeps it fast, total over any lexable input, and — crucially — able to
//! preserve comments, which the [lexer](apeiron_lexer) discards. Comments are
//! recovered from the source text that sits *between* significant tokens.

use apeiron_core::FileId;
use apeiron_lexer::{Lexer, TokenKind};

use crate::Error;

/// One logical output line: a sequence of tokens and inline comments.
#[derive(Clone, Debug, Default, PartialEq)]
struct Line {
    items: Vec<Item>,
}

impl Line {
    fn is_blank(&self) -> bool {
        self.items.is_empty()
    }
}

/// An element of an output line.
#[derive(Clone, Debug, PartialEq)]
enum Item {
    Token { kind: TokenKind, text: String },
    Comment(String),
}

impl Item {
    fn is_closer(&self) -> bool {
        matches!(
            self,
            Item::Token {
                kind: TokenKind::RBrace | TokenKind::RBracket | TokenKind::RParen,
                ..
            }
        )
    }
}

/// Rewrites `source` into its canonical form.
///
/// Formatting is idempotent: for any lexable input `s`,
/// `format_source(&format_source(s)?)? == format_source(s)?`.
///
/// # Errors
///
/// Returns the underlying [`Error::Lex`] when `source` cannot be tokenized
/// (for example an unterminated block comment or string literal).
pub fn format_source(source: &str) -> Result<String, Error> {
    let tokens = Lexer::new(source, FileId(0))
        .tokenize()
        .map_err(Error::Lex)?;

    // Drop the trailing synthetic EOF marker: it carries no text.
    let significant = &tokens[..tokens.len().saturating_sub(1)];

    let mut lines: Vec<Line> = Vec::new();
    let mut current = Line::default();
    let mut cursor = 0usize;

    for token in significant {
        let range = token.span.byte_range();
        for event in scan_trivia(&source[cursor..range.start]) {
            apply(event, &mut lines, &mut current);
        }
        current.items.push(Item::Token {
            kind: token.kind.clone(),
            text: source[range.clone()].to_string(),
        });
        cursor = range.end;
    }

    if cursor <= source.len() {
        for event in scan_trivia(&source[cursor..]) {
            apply(event, &mut lines, &mut current);
        }
    }
    lines.push(current);

    Ok(render(&lines))
}

/// A piece of trivia between two significant tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Newline,
    LineComment(String),
    BlockComment(String),
}

fn apply(event: Event, lines: &mut Vec<Line>, current: &mut Line) {
    match event {
        Event::Newline => lines.push(std::mem::take(current)),
        Event::LineComment(text) | Event::BlockComment(text) => {
            current.items.push(Item::Comment(text));
        }
    }
}

/// Splits trivia into newline and comment events in source order.
fn scan_trivia(text: &str) -> Vec<Event> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut events = Vec::new();
    let mut i = 0;
    while i < len {
        match bytes[i] {
            b'\n' => {
                events.push(Event::Newline);
                i += 1;
            }
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                let start = i;
                while i < len && bytes[i] != b'\n' {
                    i += 1;
                }
                events.push(Event::LineComment(text[start..i].to_string()));
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                let start = i;
                i += 2;
                let mut depth = 1usize;
                while i < len && depth > 0 {
                    if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
                        depth -= 1;
                        i += 2;
                    } else if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
                        depth += 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                events.push(Event::BlockComment(text[start..i].to_string()));
            }
            _ => i += 1,
        }
    }
    events
}

fn render(lines: &[Line]) -> String {
    let mut out = String::new();
    let mut depth: i32 = 0;
    let mut emitted = false;
    let mut blank_pending = false;

    for line in lines {
        if line.is_blank() {
            if emitted {
                blank_pending = true;
            }
            continue;
        }

        if blank_pending {
            out.push('\n');
            blank_pending = false;
        }

        let leading_closers = line
            .items
            .iter()
            .take_while(|item| item.is_closer())
            .count() as i32;
        let indent = (depth - leading_closers).max(0) as usize;
        out.push_str(&"    ".repeat(indent));
        out.push_str(&render_line(&line.items));
        out.push('\n');

        for item in &line.items {
            if let Item::Token { kind, .. } = item {
                match kind {
                    TokenKind::LBrace | TokenKind::LBracket | TokenKind::LParen => depth += 1,
                    TokenKind::RBrace | TokenKind::RBracket | TokenKind::RParen => depth -= 1,
                    _ => {}
                }
            }
        }

        emitted = true;
    }

    out
}

fn render_line(items: &[Item]) -> String {
    let mut out = String::new();
    let mut prev: Option<&TokenKind> = None;
    let mut prev_prefix = false;

    for item in items {
        match item {
            Item::Token { kind, text } => {
                let space = if out.is_empty() || prev_prefix {
                    false
                } else if let Some(prev) = prev {
                    needs_space(prev, kind)
                } else {
                    // No preceding token but the line is non-empty: the last
                    // element was a comment, so separate with a single space.
                    true
                };
                if space {
                    out.push(' ');
                }
                out.push_str(text);
                prev_prefix = is_prefix_operator(kind, prev);
                prev = Some(kind);
            }
            Item::Comment(text) => {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(text);
                prev = None;
                prev_prefix = false;
            }
        }
    }

    out
}

/// Whether a single space should separate `prev` and `next`.
fn needs_space(prev: &TokenKind, next: &TokenKind) -> bool {
    use TokenKind::*;

    // Never pad *before* these.
    if matches!(next, Comma | Semi | RParen | RBracket | Dot | Colon) {
        return false;
    }
    // Never pad *after* these.
    if matches!(prev, LParen | LBracket | Dot) {
        return false;
    }
    // Postfix call / index: `f(...)`, `a[...]`.
    if matches!(next, LParen | LBracket) && is_value_like(prev) {
        return false;
    }
    true
}

fn is_value_like(kind: &TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Ident(_)
            | TokenKind::Integer(_)
            | TokenKind::Float(_)
            | TokenKind::Str(_)
            | TokenKind::RParen
            | TokenKind::RBracket
    )
}

/// `+`/`-` are unary when they appear in a prefix position; `!` always is.
fn is_prefix_operator(kind: &TokenKind, prev: Option<&TokenKind>) -> bool {
    match kind {
        TokenKind::Bang => true,
        TokenKind::Plus | TokenKind::Minus => prev.map_or(true, prefix_context),
        _ => false,
    }
}

fn prefix_context(prev: &TokenKind) -> bool {
    use TokenKind::*;
    matches!(
        prev,
        LParen
            | LBracket
            | LBrace
            | Comma
            | Semi
            | Colon
            | Assign
            | Plus
            | Minus
            | Star
            | Slash
            | Percent
            | At
            | EqEq
            | NotEq
            | Lt
            | Le
            | Gt
            | Ge
            | Arrow
            | FatArrow
            | Bang
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(source: &str) -> String {
        format_source(source).expect("source should be lexable")
    }

    #[test]
    fn normalises_operator_spacing() {
        assert_eq!(fmt("let x=1+2*3"), "let x = 1 + 2 * 3\n");
    }

    #[test]
    fn keeps_calls_and_indexing_tight() {
        assert_eq!(fmt("foo ( a , b )"), "foo(a, b)\n");
        assert_eq!(fmt("a [ 0 ]"), "a[0]\n");
        assert_eq!(fmt("f()"), "f()\n");
    }

    #[test]
    fn handles_matmul_arrow_and_field_access() {
        assert_eq!(fmt("W @ x"), "W @ x\n");
        assert_eq!(fmt("cause A->B"), "cause A -> B\n");
        assert_eq!(fmt("society . Planner"), "society.Planner\n");
    }

    #[test]
    fn recognises_unary_operators() {
        assert_eq!(fmt("x = - 3"), "x = -3\n");
        assert_eq!(fmt("a * - b"), "a * -b\n");
        assert_eq!(fmt("! flag"), "!flag\n");
    }

    #[test]
    fn indents_blocks() {
        let source = "belief X {\nproposition: \"p\",\nconfidence: 0.5\n}\n";
        let expected = "belief X {\n    proposition: \"p\",\n    confidence: 0.5\n}\n";
        assert_eq!(fmt(source), expected);
    }

    #[test]
    fn outdents_closing_braces() {
        let source = "cause A {\nwhen x {\ny\n}\n}\n";
        let expected = "cause A {\n    when x {\n        y\n    }\n}\n";
        assert_eq!(fmt(source), expected);
    }

    #[test]
    fn places_else_on_the_closing_line() {
        let source = "if x {\na\n} else {\nb\n}\n";
        let expected = "if x {\n    a\n} else {\n    b\n}\n";
        assert_eq!(fmt(source), expected);
    }

    #[test]
    fn preserves_comments() {
        let source = "// header\nlet x = 1 // trailing\n/* block */ let y = 2\n";
        let expected = "// header\nlet x = 1 // trailing\n/* block */ let y = 2\n";
        assert_eq!(fmt(source), expected);
    }

    #[test]
    fn collapses_blank_lines_and_trims_edges() {
        let source = "\n\nlet a = 1\n\n\n\nlet b = 2\n\n";
        assert_eq!(fmt(source), "let a = 1\n\nlet b = 2\n");
    }

    #[test]
    fn preserves_string_contents() {
        assert_eq!(fmt("let s = \"a  b // c\""), "let s = \"a  b // c\"\n");
    }

    #[test]
    fn is_idempotent() {
        let source = "tensor W : f32[2, 3]\nlet y=W@x-1\n";
        let once = fmt(source);
        assert_eq!(fmt(&once), once);
    }

    #[test]
    fn empty_source_stays_empty() {
        assert_eq!(fmt(""), "");
        assert_eq!(fmt("   \n\n"), "");
    }

    #[test]
    fn reports_lex_errors() {
        assert!(format_source("/* unterminated").is_err());
    }
}

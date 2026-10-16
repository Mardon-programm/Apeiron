//! Converts compiler diagnostics into LSP diagnostics.

use apeiron_cli::{compile, Options};
use apeiron_core::diagnostic::{Diagnostic, Severity};
use apeiron_core::source::SourceFile;
use apeiron_core::span::Span;
use serde_json::{json, Value};

use crate::protocol::Range;

/// Compiles `source` and renders every finding as an LSP diagnostic object.
pub fn diagnostics(source: &str, uri: &str) -> Vec<Value> {
    let outcome = compile(
        source,
        uri,
        Options {
            strict: false,
            run: false,
        },
    );
    let file = outcome.map.iter().next();
    outcome
        .diagnostics
        .iter()
        .map(|diagnostic| to_lsp(diagnostic, file))
        .collect()
}

fn to_lsp(diagnostic: &Diagnostic, file: Option<&SourceFile>) -> Value {
    let range = diagnostic
        .span
        .and_then(|span| file.map(|file| range_of(span, file)))
        .map(range_json)
        .unwrap_or_else(|| range_json(Range::default()));

    let mut value = json!({
        "range": range,
        "severity": severity_code(diagnostic.severity),
        "source": "apeiron",
        "message": diagnostic.message,
    });
    if let Some(code) = &diagnostic.code {
        value["code"] = Value::String(code.clone());
    }
    value
}

fn severity_code(severity: Severity) -> u8 {
    match severity {
        Severity::Error => 1,
        Severity::Warning => 2,
        Severity::Note => 3,
    }
}

fn range_of(span: Span, file: &SourceFile) -> Range {
    let bytes = span.byte_range();
    let (start_line, start_character) = file.line_col(bytes.start);
    let (end_line, end_character) = file.line_col(bytes.end);
    Range {
        start_line: start_line.saturating_sub(1),
        start_character: start_character.saturating_sub(1),
        end_line: end_line.saturating_sub(1),
        end_character: end_character.saturating_sub(1),
    }
}

fn range_json(range: Range) -> Value {
    json!({
        "start": { "line": range.start_line, "character": range.start_character },
        "end": { "line": range.end_line, "character": range.end_character }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_source_has_no_diagnostics() {
        assert!(diagnostics("let x = 1 + 2\n", "file:///a.ape").is_empty());
    }

    #[test]
    fn parse_error_maps_to_a_range() {
        let found = diagnostics("tensor x = [1.0,\n", "file:///a.ape");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0]["severity"], 1);
        assert_eq!(found[0]["source"], "apeiron");
        assert_eq!(found[0]["code"], "PAR0001");
        assert!(found[0]["range"]["start"]["line"].is_u64());
    }
}

//! Server state and request handlers.

use std::collections::HashMap;

use apeiron_ast::builtins::{BuiltinRet, BUILTINS};
use serde_json::{json, Value};

use crate::diagnostics;
use crate::protocol::{
    DidChangeParams, DidCloseParams, DidOpenParams, FormattingParams, HoverParams, Position,
};

/// Mutable state for a single LSP session.
#[derive(Debug, Default)]
pub struct Server {
    documents: HashMap<String, String>,
    shutdown: bool,
}

impl Server {
    /// Creates an empty server.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that the client asked us to shut down.
    pub fn shut_down(&mut self) {
        self.shutdown = true;
    }

    /// Whether `shutdown` has been received.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown
    }

    /// Stores an opened document and returns its diagnostics notification.
    pub fn did_open(&mut self, params: &DidOpenParams) -> Value {
        let uri = params.text_document.uri.clone();
        self.documents
            .insert(uri.clone(), params.text_document.text.clone());
        self.publish(&uri)
    }

    /// Applies content changes (full sync) and returns the diagnostics.
    pub fn did_change(&mut self, params: &DidChangeParams) -> Value {
        let uri = params.text_document.uri.clone();
        if let Some(text) = self.documents.get_mut(&uri) {
            for change in &params.content_changes {
                *text = change.text.clone();
            }
        }
        self.publish(&uri)
    }

    /// Forgets a closed document and clears its diagnostics.
    pub fn did_close(&mut self, params: &DidCloseParams) -> Value {
        let uri = params.text_document.uri.clone();
        self.documents.remove(&uri);
        crate::protocol::publish_diagnostics(&uri, Vec::new())
    }

    /// Answers a hover request with the built-in signature, if any.
    pub fn hover(&self, params: &HoverParams) -> Value {
        let Some(text) = self.documents.get(&params.text_document.uri) else {
            return Value::Null;
        };
        let offset = offset_at(text, params.position);
        let Some(word) = word_at(text, offset) else {
            return Value::Null;
        };
        let value = match signature(word) {
            Some(signature) => format!("```apeiron\n{signature}\n```"),
            None => format!("`{word}` — Apeiron identifier"),
        };
        json!({ "contents": { "kind": "markdown", "value": value } })
    }

    /// Returns a whole-document edit that canonicalises formatting.
    pub fn formatting(&self, params: &FormattingParams) -> Value {
        let Some(text) = self.documents.get(&params.text_document.uri) else {
            return json!([]);
        };
        match apeiron_fmt::format_source(text) {
            Ok(formatted) if formatted != *text => {
                let (line, character) = end_position(text);
                json!([{
                    "range": {
                        "start": { "line": 0, "character": 0 },
                        "end": { "line": line, "character": character }
                    },
                    "newText": formatted
                }])
            }
            _ => json!([]),
        }
    }

    fn publish(&self, uri: &str) -> Value {
        let found = match self.documents.get(uri) {
            Some(text) => diagnostics::diagnostics(text, uri),
            None => Vec::new(),
        };
        crate::protocol::publish_diagnostics(uri, found)
    }
}

/// Computes the zero-based offset of `position` within `text`.
fn offset_at(text: &str, position: Position) -> usize {
    let mut offset = 0;
    for (index, line) in text.split_inclusive('\n').enumerate() {
        if index as u32 == position.line {
            let within = position.character as usize;
            return (offset + within).min(offset + line.trim_end_matches('\n').len());
        }
        offset += line.len();
    }
    text.len()
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Returns the identifier surrounding `offset`, if any.
fn word_at(text: &str, offset: usize) -> Option<&str> {
    let bytes = text.as_bytes();
    if offset > bytes.len() {
        return None;
    }
    let mut start = offset;
    while start > 0 && is_word_byte(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = offset;
    while end < bytes.len() && is_word_byte(bytes[end]) {
        end += 1;
    }
    if start == end {
        None
    } else {
        Some(&text[start..end])
    }
}

/// The signature of a built-in, if `name` names one.
fn signature(name: &str) -> Option<String> {
    let sig = BUILTINS.iter().find(|sig| sig.name == name)?;
    let params = sig
        .params
        .iter()
        .map(|param| ret_name(*param))
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!("{}({}) -> {}", sig.name, params, ret_name(sig.ret)))
}

fn ret_name(value: BuiltinRet) -> &'static str {
    match value {
        BuiltinRet::Unit => "unit",
        BuiltinRet::I64 => "i64",
        BuiltinRet::F32 => "f32",
        BuiltinRet::F64 => "f64",
        BuiltinRet::Bool => "bool",
        BuiltinRet::Tensor => "tensor",
    }
}

fn end_position(text: &str) -> (u32, u32) {
    let lines: Vec<&str> = text.split('\n').collect();
    let line = lines.len().saturating_sub(1) as u32;
    let character = lines.last().map_or(0, |last| last.chars().count()) as u32;
    (line, character)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(server: &mut Server, uri: &str, text: &str) -> Value {
        server.did_open(&DidOpenParams {
            text_document: crate::protocol::TextDocumentItem {
                uri: uri.into(),
                version: 1,
                text: text.into(),
            },
        })
    }

    #[test]
    fn publish_reports_parse_errors() {
        let mut server = Server::new();
        let note = open(&mut server, "file:///a.ape", "let x = [1.0,\n");
        assert_eq!(note["method"], "textDocument/publishDiagnostics");
        assert_eq!(note["params"]["diagnostics"][0]["code"], "PAR0001");
    }

    #[test]
    fn hover_describes_a_builtin() {
        let mut server = Server::new();
        open(&mut server, "file:///a.ape", "let y = dot(a, b)\n");
        let hover = server.hover(&HoverParams {
            text_document: crate::protocol::TextDocumentIdentifier {
                uri: "file:///a.ape".into(),
            },
            position: Position {
                line: 0,
                character: 9,
            },
        });
        let value = hover["contents"]["value"].as_str().unwrap();
        assert!(value.contains("dot(tensor, tensor) -> f32"), "{value}");
    }

    #[test]
    fn formatting_returns_a_whole_document_edit() {
        let mut server = Server::new();
        open(&mut server, "file:///a.ape", "let x=1+2\n");
        let edits = server.formatting(&FormattingParams {
            text_document: crate::protocol::TextDocumentIdentifier {
                uri: "file:///a.ape".into(),
            },
        });
        assert_eq!(edits.as_array().unwrap().len(), 1);
        assert_eq!(edits[0]["newText"], "let x = 1 + 2\n");
    }

    #[test]
    fn did_close_clears_diagnostics() {
        let mut server = Server::new();
        open(&mut server, "file:///a.ape", "let x = 1\n");
        let note = server.did_close(&DidCloseParams {
            text_document: crate::protocol::TextDocumentIdentifier {
                uri: "file:///a.ape".into(),
            },
        });
        assert_eq!(note["params"]["diagnostics"].as_array().unwrap().len(), 0);
    }
}

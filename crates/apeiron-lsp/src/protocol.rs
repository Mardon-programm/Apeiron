//! LSP protocol types: the subset the server needs.

use serde::Deserialize;
use serde_json::{json, Value};

/// A position in a document (zero-based line and UTF-16 character offset).
#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Position {
    /// Zero-based line number.
    pub line: u32,
    /// Zero-based character offset within the line.
    pub character: u32,
}

/// A half-open range in a document.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Range {
    /// Start line (zero-based).
    pub start_line: u32,
    /// Start character (zero-based).
    pub start_character: u32,
    /// End line (zero-based).
    pub end_line: u32,
    /// End character (zero-based).
    pub end_character: u32,
}

/// An unversioned document reference.
#[derive(Clone, Debug, Deserialize)]
pub struct TextDocumentIdentifier {
    /// Document URI.
    pub uri: String,
}

/// A document with its current text, as sent by `didOpen`.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextDocumentItem {
    /// Document URI.
    pub uri: String,
    /// Monotonic version.
    #[serde(default)]
    pub version: i64,
    /// Full document text.
    #[serde(default)]
    pub text: String,
}

/// Parameters for `textDocument/didOpen`.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DidOpenParams {
    /// The opened document.
    pub text_document: TextDocumentItem,
}

/// A versioned document reference.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VersionedTextDocumentIdentifier {
    /// Document URI.
    pub uri: String,
}

/// A single content change (full replacement under full sync).
#[derive(Clone, Debug, Deserialize)]
pub struct ContentChange {
    /// Replacement text.
    #[serde(default)]
    pub text: String,
}

/// Parameters for `textDocument/didChange`.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DidChangeParams {
    /// The changed document.
    pub text_document: VersionedTextDocumentIdentifier,
    /// Changes, applied in order.
    pub content_changes: Vec<ContentChange>,
}

/// Parameters for `textDocument/didClose`.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DidCloseParams {
    /// The closed document.
    pub text_document: TextDocumentIdentifier,
}

/// Parameters for `textDocument/hover`.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HoverParams {
    /// The target document.
    pub text_document: TextDocumentIdentifier,
    /// Cursor position.
    pub position: Position,
}

/// Parameters for `textDocument/formatting`.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FormattingParams {
    /// The target document.
    pub text_document: TextDocumentIdentifier,
}

/// The `initialize` result advertising our capabilities.
pub fn initialize_result() -> Value {
    json!({
        "capabilities": {
            "textDocumentSync": { "openClose": true, "change": 1 },
            "hoverProvider": true,
            "documentFormattingProvider": true
        },
        "serverInfo": {
            "name": "apeiron-lsp",
            "version": env!("CARGO_PKG_VERSION")
        }
    })
}

/// Builds a `textDocument/publishDiagnostics` notification.
pub fn publish_diagnostics(uri: &str, diagnostics: Vec<Value>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": { "uri": uri, "diagnostics": diagnostics }
    })
}

/// Builds a successful JSON-RPC response.
pub fn response(id: &Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// Builds a JSON-RPC error response.
pub fn error_response(id: &Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message }
    })
}

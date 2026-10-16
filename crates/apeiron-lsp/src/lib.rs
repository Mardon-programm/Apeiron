//! A Language Server Protocol implementation for Apeiron.
//!
//! The server speaks JSON-RPC 2.0 over stdio using [LSP
//! framing](jsonrpc::read_message). It publishes compiler diagnostics on every
//! document change, offers hover information for identifiers (including the
//! standard-library built-ins) and formats documents with [`apeiron_fmt`].
//!
//! ```
//! use apeiron_lsp::{Server, initialize_result};
//!
//! let result = initialize_result();
//! assert_eq!(result["capabilities"]["hoverProvider"], true);
//! let _ = Server::new();
//! ```

mod diagnostics;
mod jsonrpc;
mod protocol;
mod server;

use std::io::{BufRead, Write};

use serde_json::Value;

pub use jsonrpc::{read_message, write_message, Id, Message};
pub use protocol::initialize_result;
pub use server::Server;

use protocol::{DidChangeParams, DidCloseParams, DidOpenParams, FormattingParams, HoverParams};

/// Serves the LSP protocol on `input` until the client sends `exit`.
///
/// # Errors
///
/// Returns any I/O error encountered while reading or writing messages.
pub fn run<R: BufRead, W: Write>(mut input: R, mut output: W) -> std::io::Result<()> {
    let mut server = Server::new();

    while let Some(body) = read_message(&mut input)? {
        let message: Message = match serde_json::from_slice(&body) {
            Ok(message) => message,
            Err(_) => continue,
        };

        if message.is_request() {
            let id = message
                .id
                .clone()
                .and_then(|id| serde_json::to_value(id).ok())
                .unwrap_or(Value::Null);
            let method = message.method.unwrap_or_default();
            let params = message.params.unwrap_or(Value::Null);
            let response = handle_request(&mut server, &method, params, &id);
            write_message(&mut output, &response)?;
        } else if message.is_notification() {
            let method = message.method.unwrap_or_default();
            let params = message.params.unwrap_or(Value::Null);
            if let Some(note) = handle_notification(&mut server, &method, &params) {
                write_message(&mut output, &note)?;
            }
            if method == "exit" {
                break;
            }
        }
    }

    Ok(())
}

fn handle_request(server: &mut Server, method: &str, params: Value, id: &Value) -> Value {
    match method {
        "initialize" => protocol::response(id, protocol::initialize_result()),
        "shutdown" => {
            server.shut_down();
            protocol::response(id, Value::Null)
        }
        "textDocument/hover" => match serde_json::from_value::<HoverParams>(params) {
            Ok(params) => protocol::response(id, server.hover(&params)),
            Err(_) => protocol::error_response(id, -32602, "invalid hover parameters"),
        },
        "textDocument/formatting" => match serde_json::from_value::<FormattingParams>(params) {
            Ok(params) => protocol::response(id, server.formatting(&params)),
            Err(_) => protocol::error_response(id, -32602, "invalid formatting parameters"),
        },
        other => protocol::error_response(id, -32601, &format!("method not found: {other}")),
    }
}

fn handle_notification(server: &mut Server, method: &str, params: &Value) -> Option<Value> {
    match method {
        "textDocument/didOpen" => serde_json::from_value::<DidOpenParams>(params.clone())
            .ok()
            .map(|params| server.did_open(&params)),
        "textDocument/didChange" => serde_json::from_value::<DidChangeParams>(params.clone())
            .ok()
            .map(|params| server.did_change(&params)),
        "textDocument/didClose" => serde_json::from_value::<DidCloseParams>(params.clone())
            .ok()
            .map(|params| server.did_close(&params)),
        _ => None,
    }
}

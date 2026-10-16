//! End-to-end session tests for the language server.

use std::io::Cursor;

use apeiron_lsp::{read_message, run};
use serde_json::{json, Value};

fn frame(value: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(value).expect("encode message");
    let mut bytes = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
    bytes.extend(body);
    bytes
}

fn collect(output: &[u8]) -> Vec<Value> {
    let mut cursor = Cursor::new(output);
    let mut messages = Vec::new();
    while let Some(body) = read_message(&mut cursor).expect("framed message") {
        messages.push(serde_json::from_slice(&body).expect("json message"));
    }
    messages
}

fn uri(name: &str) -> String {
    format!("file:///{name}.ape")
}

#[test]
fn serves_a_full_session() {
    let mut input = Vec::new();
    input.extend(frame(
        &json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
    ));
    input.extend(frame(&json!({
        "jsonrpc":"2.0","method":"textDocument/didOpen",
        "params":{"textDocument":{"uri":uri("broken"),"languageId":"apeiron","version":1,"text":"let x = [1.0,\n"}}
    })));
    input.extend(frame(&json!({
        "jsonrpc":"2.0","method":"textDocument/didOpen",
        "params":{"textDocument":{"uri":uri("b"),"languageId":"apeiron","version":1,
            "text":"tensor a: f32[3] = [1.0, 2.0, 3.0]\ntensor b: f32[3] = [4.0, 5.0, 6.0]\nlet y = dot(a, b)\n"}}
    })));
    input.extend(frame(
        &json!({"jsonrpc":"2.0","id":2,"method":"textDocument/hover","params":{
            "textDocument":{"uri":uri("b")},"position":{"line":2,"character":9}
        }}),
    ));
    input.extend(frame(
        &json!({"jsonrpc":"2.0","id":3,"method":"textDocument/formatting","params":{
            "textDocument":{"uri":uri("b")}
        }}),
    ));
    input.extend(frame(
        &json!({"jsonrpc":"2.0","id":4,"method":"textDocument/definition","params":{}}),
    ));
    input.extend(frame(&json!({"jsonrpc":"2.0","id":5,"method":"shutdown"})));
    input.extend(frame(&json!({"jsonrpc":"2.0","method":"exit"})));

    let mut output = Vec::new();
    run(Cursor::new(input), &mut output).expect("session runs");

    let messages = collect(&output);

    // initialize response advertises our capabilities.
    let init = messages
        .iter()
        .find(|m| m["id"] == 1)
        .expect("initialize response");
    assert_eq!(init["result"]["capabilities"]["hoverProvider"], true);
    assert_eq!(
        init["result"]["capabilities"]["documentFormattingProvider"],
        true
    );

    // The broken document publishes a PAR0001 diagnostic.
    let diagnostics = messages
        .iter()
        .find(|m| {
            m["method"] == "textDocument/publishDiagnostics" && m["params"]["uri"] == uri("broken")
        })
        .expect("diagnostics notification");
    assert_eq!(diagnostics["params"]["diagnostics"][0]["code"], "PAR0001");
    assert_eq!(diagnostics["params"]["diagnostics"][0]["severity"], 1);

    // The clean document publishes no diagnostics.
    let clean = messages
        .iter()
        .find(|m| {
            m["method"] == "textDocument/publishDiagnostics" && m["params"]["uri"] == uri("b")
        })
        .expect("clean diagnostics");
    assert_eq!(clean["params"]["diagnostics"].as_array().unwrap().len(), 0);

    // Hover resolves the built-in signature.
    let hover = messages.iter().find(|m| m["id"] == 2).expect("hover");
    let value = hover["result"]["contents"]["value"].as_str().unwrap();
    assert!(value.contains("dot(tensor, tensor) -> f32"), "{value}");

    // Formatting on an already-formatted document returns no edits.
    let edits = messages.iter().find(|m| m["id"] == 3).expect("formatting");
    assert_eq!(edits["result"].as_array().unwrap().len(), 0);

    // Unknown methods answer with "method not found".
    let missing = messages.iter().find(|m| m["id"] == 4).expect("error");
    assert_eq!(missing["error"]["code"], -32601);

    // shutdown answers null.
    let shutdown = messages.iter().find(|m| m["id"] == 5).expect("shutdown");
    assert!(shutdown["result"].is_null());
}

#[test]
fn formatting_returns_an_edit_for_messy_source() {
    let mut input = Vec::new();
    input.extend(frame(
        &json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
    ));
    input.extend(frame(&json!({
        "jsonrpc":"2.0","method":"textDocument/didOpen",
        "params":{"textDocument":{"uri":uri("messy"),"languageId":"apeiron","version":1,"text":"let x=1+2\n"}}
    })));
    input.extend(frame(
        &json!({"jsonrpc":"2.0","id":2,"method":"textDocument/formatting","params":{
            "textDocument":{"uri":uri("messy")}
        }}),
    ));
    input.extend(frame(&json!({"jsonrpc":"2.0","method":"exit"})));

    let mut output = Vec::new();
    run(Cursor::new(input), &mut output).expect("session runs");

    let messages = collect(&output);
    let edits = messages.iter().find(|m| m["id"] == 2).expect("formatting");
    assert_eq!(edits["result"].as_array().unwrap().len(), 1);
    assert_eq!(edits["result"][0]["newText"], "let x = 1 + 2\n");
}

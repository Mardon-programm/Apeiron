//! Minimal JSON-RPC 2.0 transport used by the language server.
//!
//! LSP frames each message with a `Content-Length` header followed by a blank
//! line and a UTF-8 JSON body. This module implements just enough of that
//! protocol to read requests and notifications and to write responses and
//! notifications back.

use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A JSON-RPC identifier, which may be a number or a string.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Id {
    /// Numeric id.
    Number(i64),
    /// String id.
    String(String),
}

/// An incoming JSON-RPC message, before classification.
#[derive(Clone, Debug, Deserialize)]
pub struct Message {
    /// Present for requests (and for responses, which we ignore).
    #[serde(default)]
    pub id: Option<Id>,
    /// Present for requests and notifications.
    #[serde(default)]
    pub method: Option<String>,
    /// Method parameters, if any.
    #[serde(default)]
    pub params: Option<Value>,
}

impl Message {
    /// Whether this message is a request (has both a method and an id).
    pub fn is_request(&self) -> bool {
        self.method.is_some() && self.id.is_some()
    }

    /// Whether this message is a notification (has a method but no id).
    pub fn is_notification(&self) -> bool {
        self.method.is_some() && self.id.is_none()
    }
}

/// Reads one framed message, or `None` at end of input.
///
/// # Errors
///
/// Returns an error on I/O failure or a malformed header block.
pub fn read_message<R: BufRead>(reader: &mut R) -> io::Result<Option<Vec<u8>>> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            return Ok(None);
        }
        let header = line.trim_end_matches(['\r', '\n']);
        if header.is_empty() {
            break;
        }
        if let Some(value) = header.strip_prefix("Content-Length:") {
            content_length = Some(value.trim().parse().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid Content-Length")
            })?);
        }
    }

    let length = content_length
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing Content-Length"))?;
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body)?;
    Ok(Some(body))
}

/// Writes one framed message.
///
/// # Errors
///
/// Returns an error if serialization or writing fails.
pub fn write_message<W: Write>(writer: &mut W, value: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(value)?;
    write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
    writer.write_all(&body)?;
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn classifies_requests_and_notifications() {
        let request: Message =
            serde_json::from_value(json!({"id": 1, "method": "initialize"})).unwrap();
        assert!(request.is_request());
        assert!(!request.is_notification());

        let note: Message = serde_json::from_value(json!({"method": "exit"})).unwrap();
        assert!(note.is_notification());
        assert!(!note.is_request());
    }

    #[test]
    fn round_trips_a_message() {
        let mut buffer = Vec::new();
        write_message(&mut buffer, &json!({"jsonrpc": "2.0", "method": "ping"})).unwrap();
        let mut cursor = io::Cursor::new(buffer);
        let body = read_message(&mut cursor).unwrap().expect("message");
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["method"], "ping");
        assert!(read_message(&mut cursor).unwrap().is_none());
    }

    #[test]
    fn rejects_missing_content_length() {
        let mut cursor = io::Cursor::new(b"X: 1\r\n\r\n".to_vec());
        assert!(read_message(&mut cursor).is_err());
    }
}

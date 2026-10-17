//! Foundation layer of the Apeiron compiler.
//!
//! This crate owns the vocabulary shared by every other crate in the
//! workspace: byte-accurate [`Location`] / [`Span`] tracking, the
//! [`SourceMap`] used to resolve offsets back to human readable
//! `line:column` positions, and the [`Diagnostic`] model used to report
//! lexing, parsing, and analysis failures.
//!
//! The crate deliberately contains no knowledge about tokens, syntax, or
//! semantics so that it can sit at the bottom of the dependency graph.

#![deny(rustdoc::broken_intra_doc_links)]

pub mod diagnostic;
pub mod error;
pub mod ffi;
pub mod source;
pub mod span;

pub use diagnostic::{Diagnostic, DiagnosticBag, Label, Severity};
pub use error::CoreError;
pub use ffi::{AbiType, ExternDecl, ExternSig};
pub use source::{FileId, SourceFile, SourceMap};
pub use span::{Location, Span};

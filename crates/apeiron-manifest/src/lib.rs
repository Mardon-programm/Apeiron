//! Multi-file projects for Apeiron.
//!
//! A project is a directory containing an `Apeiron.toml` manifest. The manifest
//! names the entry point, optional named modules, and path dependencies; this
//! crate resolves the whole graph and can [`link`](Project::link) every source
//! file into a single program for the compiler pipeline.
//!
//! ```
//! use apeiron_manifest::parse_manifest;
//! use std::path::Path;
//!
//! let manifest = parse_manifest(Path::new("Apeiron.toml"), "name = \"demo\"\n").unwrap();
//! assert_eq!(manifest.name, "demo");
//! assert_eq!(manifest.entry, "main.ape");
//! ```

mod error;
mod manifest;
mod project;

pub use error::{Error, Result};
pub use manifest::{parse_manifest, Manifest};
pub use project::{Project, MANIFEST_FILE};

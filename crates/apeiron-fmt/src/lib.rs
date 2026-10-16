//! Canonical source formatting for Apeiron.
//!
//! [`format_source`] rewrites a program into a single canonical style: four
//! space indentation, one statement per line, normalised spacing around
//! operators and delimiters, and collapsed blank lines. Comments survive the
//! round trip, and token text (string escapes, numeric literals) is preserved
//! byte-for-byte.
//!
//! ```
//! use apeiron_fmt::format_source;
//!
//! let messy = "belief  X{proposition:\"p\",confidence:0.5}";
//! let tidy = format_source(messy).unwrap();
//! assert_eq!(tidy, "belief X { proposition: \"p\", confidence: 0.5 }\n");
//! ```

mod error;
mod format;

pub use error::Error;
pub use format::format_source;

//! Identifiers and source-anchored names.

use std::fmt;

use apeiron_core::span::Span;

/// A named entity together with the source range it came from.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Ident {
    /// Unqualified name, for example `ShortTermMem`.
    pub name: String,
    /// Source range of the identifier.
    pub span: Span,
}

impl Ident {
    /// Builds an identifier.
    pub fn new(name: impl Into<String>, span: Span) -> Self {
        Self {
            name: name.into(),
            span,
        }
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apeiron_core::source::FileId;

    #[test]
    fn display_and_span() {
        let span = Span::empty(FileId(0));
        let ident = Ident::new("Planner", span);
        assert_eq!(ident.to_string(), "Planner");
        assert_eq!(ident.span, span);
    }
}

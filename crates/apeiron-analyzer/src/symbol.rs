//! Symbol table and lexical scopes.

use std::collections::HashMap;
use std::fmt;

use apeiron_ast::{MemoryTier, TensorKind};
use apeiron_core::span::Span;

use crate::types::Type;

/// What a name was bound to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SymbolKind {
    /// `let` binding.
    Variable,
    /// `tensor` binding.
    Tensor,
    /// `parameter` binding.
    Parameter,
    /// Function parameter.
    Param,
    /// `fn` definition.
    Fn,
    /// `belief` (declared or updated).
    Belief,
    /// `distribution`.
    Distribution,
    /// Causal graph node introduced by `cause`.
    CausalNode,
    /// `memory` store.
    Memory(MemoryTier),
    /// `plan`.
    Plan,
    /// `self_model`.
    SelfModel,
    /// `architecture`.
    Architecture,
    /// `society`.
    Society,
    /// `agent` inside a society.
    Agent,
    /// `cluster` inside a society.
    Cluster,
}

impl SymbolKind {
    /// Whether the binding is writable after declaration.
    pub fn is_mutable(self) -> bool {
        matches!(self, SymbolKind::Variable | SymbolKind::Param)
    }

    /// Builds the kind for a `tensor`/`parameter` declaration.
    pub fn for_tensor(kind: TensorKind) -> SymbolKind {
        match kind {
            TensorKind::Tensor => SymbolKind::Tensor,
            TensorKind::Parameter => SymbolKind::Parameter,
        }
    }
}

impl fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            SymbolKind::Variable => "variable",
            SymbolKind::Tensor => "tensor",
            SymbolKind::Parameter => "parameter",
            SymbolKind::Param => "parameter",
            SymbolKind::Fn => "function",
            SymbolKind::Belief => "belief",
            SymbolKind::Distribution => "distribution",
            SymbolKind::CausalNode => "causal node",
            SymbolKind::Memory(_) => "memory store",
            SymbolKind::Plan => "plan",
            SymbolKind::SelfModel => "self model",
            SymbolKind::Architecture => "architecture",
            SymbolKind::Society => "society",
            SymbolKind::Agent => "agent",
            SymbolKind::Cluster => "cluster",
        };
        f.write_str(text)
    }
}

/// A resolved name.
#[derive(Clone, PartialEq, Debug)]
pub struct Symbol {
    /// Source name.
    pub name: String,
    /// What the name denotes.
    pub kind: SymbolKind,
    /// Analyzed type.
    pub ty: Type,
    /// Declaration site.
    pub span: Span,
}

impl Symbol {
    /// Builds a symbol.
    pub fn new(name: impl Into<String>, kind: SymbolKind, ty: Type, span: Span) -> Self {
        Self {
            name: name.into(),
            kind,
            ty,
            span,
        }
    }
}

/// The flavour of a scope, used to describe where names come from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScopeKind {
    /// File scope.
    Global,
    /// Function body.
    Function,
    /// Nested block.
    Block,
    /// Society body, holding `agent`/`cluster` declarations.
    Society,
}

/// A lexical scope holding bound names.
#[derive(Clone, Debug)]
pub struct Scope {
    /// Flavour of this scope.
    pub kind: ScopeKind,
    symbols: HashMap<String, Symbol>,
}

impl Scope {
    /// Creates an empty scope.
    pub fn new(kind: ScopeKind) -> Self {
        Self {
            kind,
            symbols: HashMap::new(),
        }
    }

    /// Binds `symbol`, returning the previous binding if the name existed.
    pub fn insert(&mut self, symbol: Symbol) -> Option<Symbol> {
        self.symbols.insert(symbol.name.clone(), symbol)
    }

    /// Looks a name up in this scope only.
    pub fn get(&self, name: &str) -> Option<&Symbol> {
        self.symbols.get(name)
    }

    /// Whether this scope directly binds `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.symbols.contains_key(name)
    }

    /// Number of bindings.
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Whether the scope is empty.
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// Iterates over bound symbols in unspecified order.
    pub fn symbols(&self) -> impl Iterator<Item = &Symbol> {
        self.symbols.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apeiron_core::source::FileId;

    fn span() -> Span {
        Span::empty(FileId(0))
    }

    #[test]
    fn insert_and_lookup() {
        let mut scope = Scope::new(ScopeKind::Global);
        assert!(scope.is_empty());
        let symbol = Symbol::new("x", SymbolKind::Variable, Type::I64, span());
        assert!(scope.insert(symbol).is_none());
        assert!(scope.contains("x"));
        assert_eq!(scope.get("x").unwrap().ty, Type::I64);
        assert_eq!(scope.len(), 1);
    }

    #[test]
    fn duplicate_insert_returns_previous() {
        let mut scope = Scope::new(ScopeKind::Global);
        scope.insert(Symbol::new("x", SymbolKind::Variable, Type::I64, span()));
        let previous = scope.insert(Symbol::new("x", SymbolKind::Tensor, Type::Unit, span()));
        assert_eq!(previous.unwrap().kind, SymbolKind::Variable);
        assert_eq!(scope.get("x").unwrap().kind, SymbolKind::Tensor);
    }

    #[test]
    fn mutability_and_display() {
        assert!(SymbolKind::Variable.is_mutable());
        assert!(!SymbolKind::Tensor.is_mutable());
        assert_eq!(
            SymbolKind::Memory(MemoryTier::Episodic).to_string(),
            "memory store"
        );
        assert_eq!(
            SymbolKind::for_tensor(TensorKind::Parameter),
            SymbolKind::Parameter
        );
    }
}

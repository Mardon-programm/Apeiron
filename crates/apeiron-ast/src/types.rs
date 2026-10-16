//! Type syntax: named types optionally parameterized by tensor dimensions.
//!
//! Apeiron types are written `f32[2, 3]`, `i64`, or with dimension
//! variables such as `f32[N]` for shape-generic declarations.

use std::fmt;

use apeiron_core::span::Span;

use crate::ident::Ident;

/// One dimension inside a tensor type.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TypeDim {
    /// Statically known extent, for example the `2` in `f32[2, 3]`.
    Const {
        /// Numeric extent.
        value: u64,
        /// Source range of the literal.
        span: Span,
    },
    /// Symbolic extent resolved during shape inference, for example `N`.
    Dynamic(Ident),
}

impl TypeDim {
    /// Source range of this dimension.
    pub fn span(&self) -> Span {
        match self {
            TypeDim::Const { span, .. } => *span,
            TypeDim::Dynamic(ident) => ident.span,
        }
    }
}

impl fmt::Display for TypeDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeDim::Const { value, .. } => write!(f, "{value}"),
            TypeDim::Dynamic(ident) => write!(f, "{ident}"),
        }
    }
}

/// A (possibly shaped) type expression.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TypeExpr {
    /// Base type name, for example `f32`.
    pub name: Ident,
    /// Tensor dimensions; empty for scalar types such as `i64`.
    pub dims: Vec<TypeDim>,
    /// Source range covering the whole type expression.
    pub span: Span,
}

impl TypeExpr {
    /// Builds a type expression.
    pub fn new(name: Ident, dims: Vec<TypeDim>, span: Span) -> Self {
        Self { name, dims, span }
    }

    /// A scalar type with no dimensions.
    pub fn scalar(name: Ident) -> Self {
        let span = name.span;
        Self {
            name,
            dims: Vec::new(),
            span,
        }
    }

    /// Whether this type is shaped like a tensor.
    pub fn is_tensor(&self) -> bool {
        !self.dims.is_empty()
    }
}

impl fmt::Display for TypeExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name.name)?;
        if !self.dims.is_empty() {
            f.write_str("[")?;
            for (idx, dim) in self.dims.iter().enumerate() {
                if idx > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{dim}")?;
            }
            f.write_str("]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apeiron_core::source::FileId;

    fn ident(name: &str) -> Ident {
        Ident::new(name, Span::empty(FileId(0)))
    }

    #[test]
    fn scalar_type_has_no_dims() {
        let ty = TypeExpr::scalar(ident("i64"));
        assert!(!ty.is_tensor());
        assert_eq!(ty.to_string(), "i64");
    }

    #[test]
    fn shaped_type_renders() {
        let span = Span::empty(FileId(0));
        let ty = TypeExpr::new(
            ident("f32"),
            vec![
                TypeDim::Const { value: 2, span },
                TypeDim::Const { value: 3, span },
            ],
            span,
        );
        assert!(ty.is_tensor());
        assert_eq!(ty.to_string(), "f32[2, 3]");
    }

    #[test]
    fn dynamic_dim_renders_name() {
        let span = Span::empty(FileId(0));
        let ty = TypeExpr::new(ident("f32"), vec![TypeDim::Dynamic(ident("N"))], span);
        assert_eq!(ty.to_string(), "f32[N]");
    }
}

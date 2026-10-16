//! Semantic types of the analyzer.
//!
//! The type lattice combines ordinary scalar/tensor types with opaque
//! "domain" types for cognitive entities such as beliefs, memory stores,
//! or agent societies. Two sentinel types drive error recovery:
//!
//! * [`Type::Unknown`] — the analyzer genuinely does not know the type
//!   (for example the result of an opaque field access);
//! * [`Type::Error`] — a diagnostic has already been reported, so this
//!   value must not trigger follow-up errors.
//!
//! Both sentinels are accepted anywhere, which prevents error cascades.

use std::fmt;

/// Element type of a tensor or scalar numeric value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ElemType {
    /// `i64` (and the other integer widths).
    I64,
    /// `f32`.
    F32,
    /// `f64`.
    F64,
}

impl ElemType {
    /// Canonical source spelling.
    pub const fn name(self) -> &'static str {
        match self {
            ElemType::I64 => "i64",
            ElemType::F32 => "f32",
            ElemType::F64 => "f64",
        }
    }

    /// Smallest element type that can represent both inputs.
    pub fn unify(self, other: ElemType) -> Option<ElemType> {
        if self == other {
            return Some(self);
        }
        match (self, other) {
            (ElemType::I64, ElemType::F32) | (ElemType::F32, ElemType::I64) => Some(ElemType::F32),
            (ElemType::I64, ElemType::F64) | (ElemType::F64, ElemType::I64) => Some(ElemType::F64),
            (ElemType::F32, ElemType::F64) | (ElemType::F64, ElemType::F32) => Some(ElemType::F64),
            _ => None,
        }
    }
}

impl fmt::Display for ElemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A tensor shape with `None` standing for a dimension that could not be
/// inferred statically (dynamic or wildcard).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TensorType {
    /// Element type.
    pub elem: ElemType,
    /// Dimensions from outermost to innermost.
    pub shape: Vec<Option<u64>>,
}

impl TensorType {
    /// Builds a tensor type.
    pub fn new(elem: ElemType, shape: Vec<Option<u64>>) -> Self {
        Self { elem, shape }
    }

    /// A rank-0 (scalar) tensor type.
    pub fn scalar(elem: ElemType) -> Self {
        Self::new(elem, Vec::new())
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Whether the shape is fully static.
    pub fn is_fully_static(&self) -> bool {
        self.shape.iter().all(Option::is_some)
    }

    /// Whether both shapes describe the same extents (wildcards match).
    pub fn matches(&self, other: &TensorType) -> bool {
        if self.elem.unify(other.elem).is_none() || self.rank() != other.rank() {
            return false;
        }
        self.shape
            .iter()
            .zip(&other.shape)
            .all(|(a, b)| dim_matches(*a, *b))
    }
}

fn dim_matches(a: Option<u64>, b: Option<u64>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x == y,
        _ => true,
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[", self.elem)?;
        for (idx, dim) in self.shape.iter().enumerate() {
            if idx > 0 {
                f.write_str(", ")?;
            }
            match dim {
                Some(value) => write!(f, "{value}")?,
                None => f.write_str("_")?,
            }
        }
        f.write_str("]")
    }
}

/// Opaque cognitive entity kinds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DomainKind {
    /// A `belief` value.
    Belief,
    /// A `distribution` value.
    Distribution,
    /// A node in the causal graph.
    CausalNode,
    /// A multi-tier memory store.
    MemoryStore,
    /// A `plan` block.
    Plan,
    /// A `self_model` block.
    SelfModel,
    /// An `architecture` block.
    Architecture,
    /// A `society` block.
    Society,
    /// An `agent` declared inside a society.
    Agent,
    /// A `cluster` declared inside a society.
    Cluster,
}

impl DomainKind {
    /// Human readable name used in diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            DomainKind::Belief => "belief",
            DomainKind::Distribution => "distribution",
            DomainKind::CausalNode => "causal node",
            DomainKind::MemoryStore => "memory store",
            DomainKind::Plan => "plan",
            DomainKind::SelfModel => "self model",
            DomainKind::Architecture => "architecture",
            DomainKind::Society => "society",
            DomainKind::Agent => "agent",
            DomainKind::Cluster => "cluster",
        }
    }
}

impl fmt::Display for DomainKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The analyzer type lattice.
#[derive(Clone, PartialEq, Debug)]
pub enum Type {
    /// `i64`
    I64,
    /// `f32`
    F32,
    /// `f64`
    F64,
    /// `bool`
    Bool,
    /// `str`
    Str,
    /// Absence of a value.
    Unit,
    /// Ranked tensor value.
    Tensor(TensorType),
    /// Function signature.
    Fn {
        /// Parameter types.
        params: Vec<Type>,
        /// Return type.
        ret: Box<Type>,
    },
    /// Opaque cognitive entity.
    Domain(DomainKind),
    /// Not known, but not an error.
    Unknown,
    /// A diagnostic has been reported; suppress cascades.
    Error,
}

impl Type {
    /// A scalar numeric type with the given element type.
    pub fn scalar(elem: ElemType) -> Type {
        match elem {
            ElemType::I64 => Type::I64,
            ElemType::F32 => Type::F32,
            ElemType::F64 => Type::F64,
        }
    }

    /// Whether this is `Unknown` or `Error`.
    pub fn is_sentinel(&self) -> bool {
        matches!(self, Type::Unknown | Type::Error)
    }

    /// Whether a value of this type can participate in arithmetic.
    pub fn is_numeric(&self) -> bool {
        matches!(self, Type::I64 | Type::F32 | Type::F64)
    }

    /// Numeric element type of a scalar or tensor, if any.
    pub fn numeric_elem(&self) -> Option<ElemType> {
        match self {
            Type::I64 => Some(ElemType::I64),
            Type::F32 => Some(ElemType::F32),
            Type::F64 => Some(ElemType::F64),
            Type::Tensor(tensor) => Some(tensor.elem),
            _ => None,
        }
    }

    /// Whether this is a tensor (rank > 0).
    pub fn is_tensor(&self) -> bool {
        matches!(self, Type::Tensor(tensor) if tensor.rank() > 0)
    }

    /// Whether `value` can be assigned to a slot of type `self`.
    pub fn accepts(&self, value: &Type) -> bool {
        if self.is_sentinel() || value.is_sentinel() {
            return true;
        }
        match (self, value) {
            (Type::I64, Type::I64)
            | (Type::F32, Type::F32)
            | (Type::F64, Type::F64)
            | (Type::Bool, Type::Bool)
            | (Type::Str, Type::Str)
            | (Type::Unit, Type::Unit) => true,
            // Small integer/float widening is implicit.
            (Type::F32, Type::I64) | (Type::F64, Type::I64) | (Type::F64, Type::F32) => true,
            (Type::Tensor(declared), Type::Tensor(inferred)) => declared.matches(inferred),
            (Type::Tensor(declared), Type::I64)
            | (Type::Tensor(declared), Type::F32)
            | (Type::Tensor(declared), Type::F64) => declared.rank() == 0,
            (Type::Domain(a), Type::Domain(b)) => a == b,
            (
                Type::Fn {
                    params: pa,
                    ret: ra,
                },
                Type::Fn {
                    params: pb,
                    ret: rb,
                },
            ) => pa.len() == pb.len() && pa.iter().zip(pb).all(|(a, b)| a == b) && ra == rb,
            _ => false,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::I64 => f.write_str("i64"),
            Type::F32 => f.write_str("f32"),
            Type::F64 => f.write_str("f64"),
            Type::Bool => f.write_str("bool"),
            Type::Str => f.write_str("str"),
            Type::Unit => f.write_str("()"),
            Type::Tensor(tensor) => write!(f, "{tensor}"),
            Type::Fn { params, ret } => {
                f.write_str("fn(")?;
                for (idx, param) in params.iter().enumerate() {
                    if idx > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{param}")?;
                }
                write!(f, ") -> {ret}")
            }
            Type::Domain(kind) => write!(f, "{kind}"),
            Type::Unknown => f.write_str("?"),
            Type::Error => f.write_str("<error>"),
        }
    }
}

impl TensorType {
    /// NumPy-style broadcasting of two tensor shapes.
    ///
    /// `None` dimensions act as wildcards; broadcasting a `1` against any
    /// extent yields that extent.
    pub fn broadcast(a: &TensorType, b: &TensorType) -> Option<TensorType> {
        let elem = a.elem.unify(b.elem)?;
        let rank = a.rank().max(b.rank());
        let mut shape = Vec::with_capacity(rank);
        for position in 0..rank {
            let dim_a = dim_from_right(&a.shape, position);
            let dim_b = dim_from_right(&b.shape, position);
            let dim = match (dim_a, dim_b) {
                (None, Some(dim)) | (Some(dim), None) => dim,
                (Some(x), Some(y)) => match (x, y) {
                    (None, _) | (_, None) => None,
                    (Some(1), Some(v)) | (Some(v), Some(1)) => Some(v),
                    (Some(x), Some(y)) if x == y => Some(x),
                    _ => return None,
                },
                (None, None) => unreachable!("position is below max rank"),
            };
            shape.push(dim);
        }
        shape.reverse();
        Some(TensorType::new(elem, shape))
    }

    /// Matrix multiplication shape rule covering matrix/vector products.
    pub fn matmul(a: &TensorType, b: &TensorType) -> Option<TensorType> {
        let elem = a.elem.unify(b.elem)?;
        let shape = match (a.rank(), b.rank()) {
            (2, 2) => {
                inner_dim(a.shape[1], b.shape[0])?;
                vec![a.shape[0], b.shape[1]]
            }
            (2, 1) => {
                inner_dim(a.shape[1], b.shape[0])?;
                vec![a.shape[0]]
            }
            (1, 2) => {
                inner_dim(a.shape[0], b.shape[0])?;
                vec![b.shape[1]]
            }
            (1, 1) => {
                inner_dim(a.shape[0], b.shape[0])?;
                Vec::new()
            }
            _ => return None,
        };
        Some(TensorType::new(elem, shape))
    }
}

fn dim_from_right(shape: &[Option<u64>], position: usize) -> Option<Option<u64>> {
    shape.len().checked_sub(position + 1).map(|idx| shape[idx])
}

fn inner_dim(a: Option<u64>, b: Option<u64>) -> Option<()> {
    match (a, b) {
        (Some(x), Some(y)) if x != y => None,
        _ => Some(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(dims: &[Option<u64>]) -> TensorType {
        TensorType::new(ElemType::F32, dims.to_vec())
    }

    #[test]
    fn elem_unification_orders_precision() {
        assert_eq!(ElemType::I64.unify(ElemType::F32), Some(ElemType::F32));
        assert_eq!(ElemType::F32.unify(ElemType::F64), Some(ElemType::F64));
        assert_eq!(ElemType::I64.unify(ElemType::I64), Some(ElemType::I64));
    }

    #[test]
    fn display_shapes() {
        assert_eq!(shape(&[Some(2), Some(3)]).to_string(), "f32[2, 3]");
        assert_eq!(shape(&[Some(2), None]).to_string(), "f32[2, _]");
    }

    #[test]
    fn matches_allows_wildcards() {
        assert!(shape(&[Some(2), Some(3)]).matches(&shape(&[Some(2), Some(3)])));
        assert!(shape(&[Some(2), None]).matches(&shape(&[Some(2), Some(3)])));
        assert!(!shape(&[Some(2), Some(3)]).matches(&shape(&[Some(4), Some(3)])));
        assert!(!shape(&[Some(2)]).matches(&shape(&[Some(2), Some(3)])));
    }

    #[test]
    fn broadcast_rules() {
        let a = shape(&[Some(2), Some(3)]);
        let b = shape(&[Some(1), Some(3)]);
        assert_eq!(
            TensorType::broadcast(&a, &b).unwrap().shape,
            vec![Some(2), Some(3)]
        );
        let c = shape(&[Some(3)]);
        assert_eq!(
            TensorType::broadcast(&a, &c).unwrap().shape,
            vec![Some(2), Some(3)]
        );
        let bad = shape(&[Some(4)]);
        assert!(TensorType::broadcast(&a, &bad).is_none());
    }

    #[test]
    fn matmul_shapes() {
        let m = shape(&[Some(2), Some(3)]);
        let v = shape(&[Some(3)]);
        assert_eq!(TensorType::matmul(&m, &v).unwrap().shape, vec![Some(2)]);
        let n = shape(&[Some(3), Some(4)]);
        assert_eq!(
            TensorType::matmul(&m, &n).unwrap().shape,
            vec![Some(2), Some(4)]
        );
        let bad = shape(&[Some(5)]);
        assert!(TensorType::matmul(&m, &bad).is_none());
    }

    #[test]
    fn accepts_numeric_widening() {
        assert!(Type::F32.accepts(&Type::I64));
        assert!(!Type::I64.accepts(&Type::F32));
        assert!(Type::Tensor(TensorType::new(ElemType::F32, vec![Some(3)]))
            .accepts(&Type::Tensor(TensorType::new(ElemType::I64, vec![Some(3)]))));
    }

    #[test]
    fn error_and_unknown_accept_everything() {
        assert!(Type::Error.accepts(&Type::Str));
        assert!(Type::Str.accepts(&Type::Unknown));
    }
}

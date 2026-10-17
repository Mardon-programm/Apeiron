//! Types of the Apeiron intermediate representation.
//!
//! The IR keeps a small, explicit type system: scalars, ranked tensors
//! with static-or-dynamic dimensions, and opaque *domain* types for the
//! cognitive entities. Keeping this separate from the analyzer's type
//! lattice means the backend never has to reason about inference
//! sentinels (`?`, `<error>`).

use std::fmt;

use apeiron_ast::MemoryTier;

/// Scalar element of a tensor.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ElemType {
    /// 64-bit signed integer.
    I64,
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
}

impl ElemType {
    /// Canonical spelling.
    pub const fn name(self) -> &'static str {
        match self {
            ElemType::I64 => "i64",
            ElemType::F32 => "f32",
            ElemType::F64 => "f64",
        }
    }

    /// Size in bytes of one element.
    pub const fn byte_size(self) -> usize {
        match self {
            ElemType::I64 => 8,
            ElemType::F32 => 4,
            ElemType::F64 => 8,
        }
    }
}

impl fmt::Display for ElemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A single tensor dimension.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Dim {
    /// Statically known extent.
    Const(u64),
    /// Extent only known at run time.
    Dynamic,
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Const(value) => write!(f, "{value}"),
            Dim::Dynamic => f.write_str("?"),
        }
    }
}

/// A ranked tensor type.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TensorTy {
    /// Element type.
    pub elem: ElemType,
    /// Dimensions from outermost to innermost.
    pub shape: Vec<Dim>,
}

impl TensorTy {
    /// Builds a tensor type.
    pub fn new(elem: ElemType, shape: Vec<Dim>) -> Self {
        Self { elem, shape }
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements when every dimension is static.
    pub fn element_count(&self) -> Option<u64> {
        let mut total = 1u64;
        for dim in &self.shape {
            match dim {
                Dim::Const(value) => total *= value,
                Dim::Dynamic => return None,
            }
        }
        Some(total)
    }

    /// Whether every dimension is statically known.
    pub fn is_fully_static(&self) -> bool {
        self.shape.iter().all(|dim| matches!(dim, Dim::Const(_)))
    }
}

impl fmt::Display for TensorTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[", self.elem)?;
        for (index, dim) in self.shape.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{dim}")?;
        }
        f.write_str("]")
    }
}

/// Opaque cognitive entity carried through the IR.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DomainKind {
    /// A Bayesian belief.
    Belief,
    /// A categorical distribution.
    Distribution,
    /// A continuous node in the causal graph.
    CausalNode,
    /// A tiered memory store.
    Memory(MemoryTier),
    /// A plan.
    Plan,
    /// A self model.
    SelfModel,
    /// A runtime architecture.
    Architecture,
    /// An agent society.
    Society,
    /// An agent inside a society.
    Agent,
    /// A cluster inside a society.
    Cluster,
}

impl DomainKind {
    /// Human readable name.
    pub const fn name(self) -> &'static str {
        match self {
            DomainKind::Belief => "belief",
            DomainKind::Distribution => "distribution",
            DomainKind::CausalNode => "causal node",
            DomainKind::Memory(_) => "memory",
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
        match self {
            DomainKind::Memory(tier) => write!(f, "memory<{tier}>"),
            other => f.write_str(other.name()),
        }
    }
}

/// An `Option`/`Result` enum type.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct EnumTy {
    /// Enum type name (`Option` or `Result`).
    pub name: String,
    /// Payload types, in variant argument order (`Option` has one,
    /// `Result` has two: success then error).
    pub payload: Vec<IrType>,
}

impl EnumTy {
    /// Builds an enum type with its payload types.
    pub fn new(name: impl Into<String>, payload: Vec<IrType>) -> Self {
        Self {
            name: name.into(),
            payload,
        }
    }

    /// The payload type at `index`, if declared.
    pub fn payload_at(&self, index: usize) -> Option<&IrType> {
        self.payload.get(index)
    }
}

impl fmt::Display for EnumTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.payload.is_empty() {
            return f.write_str(&self.name);
        }
        write!(f, "{}[", self.name)?;
        for (index, ty) in self.payload.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{ty}")?;
        }
        f.write_str("]")
    }
}

/// A fully described IR type.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum IrType {
    /// No value.
    Unit,
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
    /// Not yet resolved: an enum payload slot whose type the surrounding
    /// expression does not pin down (e.g. the error half of `Err(x)`).
    Unknown,
    /// Ranked tensor.
    Tensor(TensorTy),
    /// `Option`/`Result` enum.
    Enum(EnumTy),
    /// Opaque cognitive entity.
    Domain(DomainKind),
}

impl IrType {
    /// A scalar type with the given element type.
    pub const fn scalar(elem: ElemType) -> IrType {
        match elem {
            ElemType::I64 => IrType::I64,
            ElemType::F32 => IrType::F32,
            ElemType::F64 => IrType::F64,
        }
    }

    /// Whether this is a numeric scalar.
    pub fn is_numeric(&self) -> bool {
        matches!(self, IrType::I64 | IrType::F32 | IrType::F64)
    }

    /// Whether this is a tensor of rank > 0.
    pub fn is_tensor(&self) -> bool {
        matches!(self, IrType::Tensor(tensor) if tensor.rank() > 0)
    }
}

impl fmt::Display for IrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrType::Unit => f.write_str("()"),
            IrType::I64 => f.write_str("i64"),
            IrType::F32 => f.write_str("f32"),
            IrType::F64 => f.write_str("f64"),
            IrType::Bool => f.write_str("bool"),
            IrType::Str => f.write_str("str"),
            IrType::Unknown => f.write_str("?"),
            IrType::Tensor(tensor) => write!(f, "{tensor}"),
            IrType::Enum(enum_ty) => write!(f, "{enum_ty}"),
            IrType::Domain(kind) => write!(f, "{kind}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_display_and_element_count() {
        let tensor = TensorTy::new(ElemType::F32, vec![Dim::Const(2), Dim::Const(3)]);
        assert_eq!(tensor.to_string(), "f32[2, 3]");
        assert_eq!(tensor.element_count(), Some(6));
        assert!(tensor.is_fully_static());
    }

    #[test]
    fn dynamic_dimension_blocks_element_count() {
        let tensor = TensorTy::new(ElemType::F64, vec![Dim::Const(4), Dim::Dynamic]);
        assert_eq!(tensor.to_string(), "f64[4, ?]");
        assert_eq!(tensor.element_count(), None);
        assert!(!tensor.is_fully_static());
    }

    #[test]
    fn domain_kind_names_memory_tier() {
        assert_eq!(
            DomainKind::Memory(MemoryTier::Semantic).to_string(),
            "memory<semantic>"
        );
        assert_eq!(DomainKind::CausalNode.name(), "causal node");
    }

    #[test]
    fn scalar_type_helper() {
        assert_eq!(IrType::scalar(ElemType::I64), IrType::I64);
        assert!(IrType::I64.is_numeric());
        assert!(!IrType::Bool.is_numeric());
    }
}

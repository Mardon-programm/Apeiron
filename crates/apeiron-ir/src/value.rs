//! SSA value identities.
//!
//! Every SSA value is identified by a [`ValueId`] and is defined exactly
//! once, either as a block parameter or as the result of an instruction.
//! Function/block references use their own newtypes so the verifier can
//! never confuse a value index with a block index.

use crate::types::IrType;

/// Dense index of a value inside a [`Function`](crate::function::Function).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ValueId(pub u32);

/// Dense index of a basic block inside a function.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct BlockId(pub u32);

/// Dense index of a function inside a module.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct FunctionId(pub u32);

impl ValueId {
    /// Display form used by the textual printer (`%0`).
    pub fn display(self) -> String {
        format!("%{}", self.0)
    }
}

impl std::fmt::Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// How a value came into existence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ValueDef {
    /// A block parameter (function arguments are the entry block's).
    Param {
        /// Owning block.
        block: BlockId,
        /// Position within the block's parameter list.
        index: usize,
    },
    /// The result of an instruction.
    Inst {
        /// Owning block.
        block: BlockId,
        /// Position of the instruction within the block.
        index: usize,
    },
}

/// An SSA value: its identity, type, and definition site.
#[derive(Clone, Debug)]
pub struct Value {
    /// Identity.
    pub id: ValueId,
    /// Static type.
    pub ty: IrType,
    /// Definition site.
    pub def: ValueDef,
}

impl Value {
    /// Builds a value.
    pub fn new(id: ValueId, ty: IrType, def: ValueDef) -> Self {
        Self { id, ty, def }
    }

    /// The block that defines this value.
    pub fn block(&self) -> BlockId {
        match self.def {
            ValueDef::Param { block, .. } | ValueDef::Inst { block, .. } => block,
        }
    }

    /// Whether this value is a block parameter.
    pub fn is_param(&self) -> bool {
        matches!(self.def, ValueDef::Param { .. })
    }

    /// Whether this value is an instruction result.
    pub fn is_inst(&self) -> bool {
        matches!(self.def, ValueDef::Inst { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_forms() {
        assert_eq!(ValueId(3).to_string(), "%3");
        assert_eq!(ValueId(3).display(), "%3");
        assert_eq!(BlockId(2).to_string(), "bb2");
    }

    #[test]
    fn value_def_classification() {
        let value = Value::new(
            ValueId(0),
            IrType::I64,
            ValueDef::Param {
                block: BlockId(0),
                index: 0,
            },
        );
        assert!(value.is_param());
        assert!(!value.is_inst());
        assert_eq!(value.block(), BlockId(0));
    }
}

//! Modules, functions, basic blocks, and terminators.
//!
//! The IR follows an MLIR-style SSA form where control flow merges carry
//! *block parameters* instead of phi nodes: a branch passes values to the
//! target block, which binds them to its parameters.

use apeiron_core::span::Span;

use crate::instr::{ConstValue, Instruction, TensorLit};
use crate::types::{DomainKind, IrType};
use crate::value::{BlockId, FunctionId, Value, ValueId};

/// How a basic block ends.
#[derive(Clone, Debug)]
pub enum Terminator {
    /// Return, optionally with a value.
    Ret(Option<ValueId>),
    /// Unconditional branch that binds arguments to the target's params.
    Jump {
        /// Destination block.
        target: BlockId,
        /// Values bound to the target's parameters.
        args: Vec<ValueId>,
    },
    /// Conditional branch.
    Branch {
        /// Boolean condition.
        condition: ValueId,
        /// Block taken when the condition is true.
        then_block: BlockId,
        /// Arguments for the true branch.
        then_args: Vec<ValueId>,
        /// Block taken otherwise.
        else_block: BlockId,
        /// Arguments for the false branch.
        else_args: Vec<ValueId>,
    },
    /// Marks a block that cannot be reached or does not continue.
    Unreachable,
}

impl Terminator {
    /// The blocks this terminator transfers control to.
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Terminator::Ret(_) | Terminator::Unreachable => Vec::new(),
            Terminator::Jump { target, .. } => vec![*target],
            Terminator::Branch {
                then_block,
                else_block,
                ..
            } => vec![*then_block, *else_block],
        }
    }

    /// The values consumed by this terminator.
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            Terminator::Ret(value) => value.iter().copied().collect(),
            Terminator::Jump { args, .. } => args.clone(),
            Terminator::Branch {
                condition,
                then_args,
                else_args,
                ..
            } => {
                let mut operands = vec![*condition];
                operands.extend(then_args.iter().copied());
                operands.extend(else_args.iter().copied());
                operands
            }
            Terminator::Unreachable => Vec::new(),
        }
    }
}

/// A basic block: parameters, instructions, and a terminator.
#[derive(Clone, Debug)]
pub struct BasicBlock {
    /// Identity.
    pub id: BlockId,
    /// Human readable label.
    pub label: String,
    /// SSA parameters bound by incoming branches.
    pub params: Vec<ValueId>,
    /// Instructions in order.
    pub instructions: Vec<Instruction>,
    /// Terminator; absent only while still being built.
    pub terminator: Option<Terminator>,
}

impl BasicBlock {
    /// Builds an empty block.
    pub fn new(id: BlockId, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            params: Vec::new(),
            instructions: Vec::new(),
            terminator: None,
        }
    }
}

/// A lowered function.
#[derive(Clone, Debug)]
pub struct Function {
    /// Identity within the module.
    pub id: FunctionId,
    /// Function name.
    pub name: String,
    /// Ordered parameter values (the entry block's parameters).
    pub params: Vec<ValueId>,
    /// Return type.
    pub return_type: IrType,
    /// All values defined in the function.
    pub values: Vec<Value>,
    /// All basic blocks.
    pub blocks: Vec<BasicBlock>,
    /// Entry block.
    pub entry: BlockId,
    /// Source range of the definition.
    pub span: Option<Span>,
}

impl Function {
    /// Looks up a block by id.
    pub fn block(&self, id: BlockId) -> &BasicBlock {
        &self.blocks[id.0 as usize]
    }

    /// Looks up a block by id, mutably.
    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock {
        &mut self.blocks[id.0 as usize]
    }

    /// Looks up a value by id.
    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id.0 as usize]
    }

    /// Type of a value.
    pub fn value_type(&self, id: ValueId) -> &IrType {
        &self.value(id).ty
    }

    /// The entry block.
    pub fn entry_block(&self) -> &BasicBlock {
        self.block(self.entry)
    }

    /// Iterates over every block paired with its id.
    pub fn blocks_with_ids(&self) -> impl Iterator<Item = (BlockId, &BasicBlock)> {
        self.blocks.iter().map(|block| (block.id, block))
    }

    /// Number of basic blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

/// What a global symbol denotes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GlobalKind {
    /// A `tensor`.
    Tensor,
    /// A trainable `parameter`.
    Parameter,
    /// A `memory` store.
    Memory,
    /// A `cause` graph node.
    CausalNode,
    /// A `belief`.
    Belief,
    /// A `distribution`.
    Distribution,
    /// A `plan`.
    Plan,
    /// A `self_model`.
    SelfModel,
    /// An `architecture`.
    Architecture,
    /// A `society`.
    Society,
}

/// The initial value of a global, when it is a compile-time constant.
#[derive(Clone, Debug)]
pub enum GlobalInit {
    /// Scalar constant.
    Const(ConstValue),
    /// Tensor literal.
    Tensor(TensorLit),
}

/// A module-level symbol.
#[derive(Clone, Debug)]
pub struct Global {
    /// Name.
    pub name: String,
    /// Type.
    pub ty: IrType,
    /// Category.
    pub kind: GlobalKind,
    /// Optional constant initializer.
    pub init: Option<GlobalInit>,
    /// Source range.
    pub span: Option<Span>,
}

impl Global {
    /// Builds a global.
    pub fn new(name: impl Into<String>, ty: IrType, kind: GlobalKind) -> Self {
        Self {
            name: name.into(),
            ty,
            kind,
            init: None,
            span: None,
        }
    }

    /// Attaches a constant initializer.
    pub fn with_init(mut self, init: GlobalInit) -> Self {
        self.init = Some(init);
        self
    }

    /// Attaches a source range.
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Convenience constructor for a memory store global.
    pub fn memory(name: impl Into<String>, tier: apeiron_ast::MemoryTier) -> Self {
        Global::new(
            name,
            IrType::Domain(DomainKind::Memory(tier)),
            GlobalKind::Memory,
        )
    }
}

/// A compiled module.
#[derive(Clone, Debug)]
pub struct Module {
    /// Module name.
    pub name: String,
    /// Global symbols.
    pub globals: Vec<Global>,
    /// Functions.
    pub functions: Vec<Function>,
}

impl Module {
    /// Builds an empty module.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            globals: Vec::new(),
            functions: Vec::new(),
        }
    }

    /// Finds a function by name.
    pub fn function(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|function| function.name == name)
    }

    /// Finds a global by name.
    pub fn global(&self, name: &str) -> Option<&Global> {
        self.globals.iter().find(|global| global.name == name)
    }

    /// Number of instructions across all functions.
    pub fn instruction_count(&self) -> usize {
        self.functions
            .iter()
            .flat_map(|function| &function.blocks)
            .map(|block| block.instructions.len())
            .sum()
    }
}

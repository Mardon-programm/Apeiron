//! The Apeiron intermediate representation.
//!
//! This crate turns the parsed [`Program`](apeiron_ast::Program) into a
//! typed SSA module and provides the machinery every backend needs:
//!
//! * [`lower_program`] walks the AST and builds a [`Module`];
//! * [`verify_module`] checks structural and SSA invariants;
//! * [`PassManager`] runs optimisation passes such as constant folding and
//!   dead-code elimination;
//! * [`print_module`] renders the IR as text for debugging.
//!
//! # SSA form
//!
//! Control-flow merges carry *block parameters* rather than phi nodes: a
//! branch passes values and the target block binds them to its parameters.
//! Function arguments are simply the entry block's parameters.

mod builder;
mod function;
mod instr;
mod lower;
mod passes;
mod print;
mod types;
mod value;
mod verify;

pub use builder::FunctionBuilder;
pub use function::{BasicBlock, Function, Global, GlobalInit, GlobalKind, Module, Terminator};
pub use instr::{
    ArchInstr, BeliefInstr, CausalInstr, ConstValue, InstrKind, Instruction, MemoryInstr,
    SocietyInstr, TensorLit,
};
pub use lower::{lower_program, LowerError};
pub use passes::{ConstantFolding, DeadCodeElimination, Pass, PassManager};
pub use print::{print_function, print_module};
pub use types::{Dim, DomainKind, ElemType, IrType, TensorTy};
pub use value::{BlockId, FunctionId, Value, ValueDef, ValueId};
pub use verify::{verify_function, verify_module, VerifyError, VerifyErrorKind};

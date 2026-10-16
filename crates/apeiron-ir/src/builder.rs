//! A cursor-based builder for constructing a [`Function`].
//!
//! The builder owns a partially built function, tracks the *current*
//! insertion block, and hands back a [`ValueId`] for every emitting call.
//! Block parameters are created eagerly so that branches can bind their
//! arguments without a separate fix-up pass.

use apeiron_core::span::Span;

use crate::function::{BasicBlock, Function, Terminator};
use crate::instr::{ConstValue, InstrKind, Instruction};
use crate::types::{ElemType, IrType};
use crate::value::{BlockId, FunctionId, Value, ValueDef, ValueId};

/// Incrementally builds a [`Function`].
pub struct FunctionBuilder {
    function: Function,
    current: BlockId,
}

impl FunctionBuilder {
    /// Starts a function with the given parameter and return types.
    pub fn new(name: impl Into<String>, param_types: &[IrType], return_type: IrType) -> Self {
        let entry = BlockId(0);
        let mut function = Function {
            id: FunctionId(0),
            name: name.into(),
            params: Vec::new(),
            return_type,
            values: Vec::new(),
            blocks: vec![BasicBlock::new(entry, "entry")],
            entry,
            span: None,
        };
        let mut params = Vec::with_capacity(param_types.len());
        for (index, ty) in param_types.iter().enumerate() {
            let value = push_value(
                &mut function,
                ty.clone(),
                ValueDef::Param {
                    block: entry,
                    index,
                },
            );
            function.blocks[0].params.push(value);
            params.push(value);
        }
        function.params = params;
        Self {
            function,
            current: entry,
        }
    }

    /// Attaches a source range to the function.
    pub fn with_span(mut self, span: Span) -> Self {
        self.function.span = Some(span);
        self
    }

    /// Sets the function identity (assigned when inserted into a module).
    pub fn set_id(&mut self, id: FunctionId) {
        self.function.id = id;
    }

    /// The entry block.
    pub fn entry(&self) -> BlockId {
        self.function.entry
    }

    /// The block new instructions are appended to.
    pub fn current_block(&self) -> BlockId {
        self.current
    }

    /// The `index`-th function parameter.
    pub fn param(&self, index: usize) -> ValueId {
        self.function.params[index]
    }

    /// All function parameters.
    pub fn params(&self) -> &[ValueId] {
        &self.function.params
    }

    /// Parameters of a block.
    pub fn block_params(&self, block: BlockId) -> &[ValueId] {
        &self.function.block(block).params
    }

    /// Type of a value.
    pub fn value_type(&self, value: ValueId) -> &IrType {
        self.function.value_type(value)
    }

    /// Whether the current block already has a terminator.
    pub fn is_terminated(&self) -> bool {
        self.function.block(self.current).terminator.is_some()
    }

    /// Creates an empty block and returns its id.
    pub fn create_block(&mut self, label: impl Into<String>) -> BlockId {
        self.create_block_with_params(label, &[])
    }

    /// Creates a block with typed parameters and returns its id.
    pub fn create_block_with_params(
        &mut self,
        label: impl Into<String>,
        param_types: &[IrType],
    ) -> BlockId {
        let id = BlockId(self.function.blocks.len() as u32);
        let mut block = BasicBlock::new(id, label);
        for (index, ty) in param_types.iter().enumerate() {
            let value = push_value(
                &mut self.function,
                ty.clone(),
                ValueDef::Param { block: id, index },
            );
            block.params.push(value);
        }
        self.function.blocks.push(block);
        id
    }

    /// Moves the insertion point to `block`.
    pub fn switch_to(&mut self, block: BlockId) {
        debug_assert!(
            (block.0 as usize) < self.function.blocks.len(),
            "unknown block"
        );
        self.current = block;
    }

    /// Emits an instruction that yields a value, returning the new value.
    pub fn emit(&mut self, kind: InstrKind, ty: IrType, span: Option<Span>) -> ValueId {
        debug_assert!(!self.is_terminated(), "cannot emit past a terminator");
        let block = self.current;
        let index = self.function.block(block).instructions.len();
        let result = push_value(&mut self.function, ty, ValueDef::Inst { block, index });
        let instruction = Instruction {
            result: Some(result),
            kind,
            span,
        };
        self.function
            .block_mut(block)
            .instructions
            .push(instruction);
        result
    }

    /// Emits an effect-only instruction.
    pub fn emit_effect(&mut self, kind: InstrKind, span: Option<Span>) {
        debug_assert!(!self.is_terminated(), "cannot emit past a terminator");
        let instruction = Instruction {
            result: None,
            kind,
            span,
        };
        self.function
            .block_mut(self.current)
            .instructions
            .push(instruction);
    }

    /// Emits a constant instruction.
    pub fn emit_const(&mut self, value: ConstValue, ty: IrType) -> ValueId {
        self.emit(InstrKind::Const(value), ty, None)
    }

    /// Emits an integer constant.
    pub fn const_i64(&mut self, value: i64) -> ValueId {
        self.emit_const(ConstValue::Int(value), IrType::I64)
    }

    /// Emits a float constant of the requested element type.
    pub fn const_float(&mut self, value: f64, elem: ElemType) -> ValueId {
        self.emit_const(ConstValue::Float(value), IrType::scalar(elem))
    }

    /// Emits a string constant.
    pub fn const_str(&mut self, value: impl Into<String>) -> ValueId {
        self.emit_const(ConstValue::Str(value.into()), IrType::Str)
    }

    /// Sets the terminator of the current block.
    pub fn terminate(&mut self, terminator: Terminator) {
        let block = self.current;
        debug_assert!(
            self.function.block(block).terminator.is_none(),
            "block already terminated"
        );
        self.function.block_mut(block).terminator = Some(terminator);
    }

    /// Borrows the function under construction.
    pub fn function(&self) -> &Function {
        &self.function
    }

    /// Consumes the builder.
    pub fn build(self) -> Function {
        self.function
    }
}

fn push_value(function: &mut Function, ty: IrType, def: ValueDef) -> ValueId {
    let id = ValueId(function.values.len() as u32);
    function.values.push(Value::new(id, ty, def));
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_a_simple_function() {
        let mut builder = FunctionBuilder::new("add", &[IrType::I64, IrType::I64], IrType::I64);
        assert_eq!(builder.params().len(), 2);
        let lhs = builder.param(0);
        let rhs = builder.param(1);
        let sum = builder.emit(
            InstrKind::Binary {
                op: apeiron_ast::BinaryOp::Add,
                lhs,
                rhs,
            },
            IrType::I64,
            None,
        );
        builder.terminate(Terminator::Ret(Some(sum)));

        let function = builder.build();
        assert_eq!(function.name, "add");
        assert_eq!(function.block_count(), 1);
        assert_eq!(function.entry_block().instructions.len(), 1);
        assert!(matches!(
            function.entry_block().terminator,
            Some(Terminator::Ret(Some(_)))
        ));
    }

    #[test]
    fn block_parameters_are_values() {
        let mut builder = FunctionBuilder::new("f", &[], IrType::I64);
        let merge = builder.create_block_with_params("merge", &[IrType::I64]);
        assert_eq!(builder.block_params(merge).len(), 1);
        let param = builder.block_params(merge)[0];
        assert_eq!(builder.value_type(param), &IrType::I64);
        assert!(builder.function().value(param).is_param());
    }
}

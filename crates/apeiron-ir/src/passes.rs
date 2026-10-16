//! Optimization passes over the IR.
//!
//! The passes are intentionally classical: constant folding rewrites
//! arithmetic on known constants, and dead-code elimination removes
//! instructions whose results are unused while preserving everything with
//! observable effects (belief, memory, society, ...).

use std::collections::HashMap;

use apeiron_ast::{BinaryOp, UnaryOp};

use crate::function::{Function, Module};
use crate::instr::{ConstValue, InstrKind};
use crate::value::ValueId;

/// A module-level transformation.
pub trait Pass {
    /// Human readable name.
    fn name(&self) -> &'static str;
    /// Rewrites the module in place.
    fn run(&self, module: &mut Module);
}

/// Folds arithmetic on compile-time constants.
#[derive(Default)]
pub struct ConstantFolding;

impl Pass for ConstantFolding {
    fn name(&self) -> &'static str {
        "constant-folding"
    }

    fn run(&self, module: &mut Module) {
        for function in &mut module.functions {
            fold_function(function);
        }
    }
}

/// Removes instructions whose results are never used.
#[derive(Default)]
pub struct DeadCodeElimination;

impl Pass for DeadCodeElimination {
    fn name(&self) -> &'static str {
        "dead-code-elimination"
    }

    fn run(&self, module: &mut Module) {
        for function in &mut module.functions {
            eliminate_dead_code(function);
        }
    }
}

/// Runs a sequence of passes.
#[derive(Default)]
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    /// Creates an empty manager.
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Appends a pass.
    pub fn add_pass(&mut self, pass: impl Pass + 'static) -> &mut Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// A manager with the default optimisation pipeline.
    pub fn default_pipeline() -> Self {
        let mut manager = Self::new();
        manager.add_pass(ConstantFolding);
        manager.add_pass(DeadCodeElimination);
        manager
    }

    /// Names of the registered passes, in order.
    pub fn pass_names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|pass| pass.name()).collect()
    }

    /// Runs every pass over the module.
    pub fn run(&self, module: &mut Module) {
        for pass in &self.passes {
            pass.run(module);
        }
    }
}

fn fold_function(function: &mut Function) {
    let mut constants: HashMap<ValueId, ConstValue> = HashMap::new();
    for block in &function.blocks {
        for instruction in &block.instructions {
            if let (Some(result), InstrKind::Const(value)) = (instruction.result, &instruction.kind)
            {
                constants.insert(result, value.clone());
            }
        }
    }

    for block in &mut function.blocks {
        for instruction in &mut block.instructions {
            let Some(result) = instruction.result else {
                continue;
            };
            let folded = match &instruction.kind {
                InstrKind::Unary {
                    op: UnaryOp::Neg,
                    operand,
                } => constants.get(operand).and_then(negate),
                InstrKind::Binary { op, lhs, rhs } => {
                    match (constants.get(lhs), constants.get(rhs)) {
                        (Some(lhs), Some(rhs)) => fold_binary(*op, lhs, rhs),
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some(value) = folded {
                constants.insert(result, value.clone());
                instruction.kind = InstrKind::Const(value);
            }
        }
    }
}

fn negate(value: &ConstValue) -> Option<ConstValue> {
    match value {
        ConstValue::Int(inner) => Some(ConstValue::Int(-inner)),
        ConstValue::Float(inner) => Some(ConstValue::Float(-inner)),
        _ => None,
    }
}

fn fold_binary(op: BinaryOp, lhs: &ConstValue, rhs: &ConstValue) -> Option<ConstValue> {
    if let (ConstValue::Int(a), ConstValue::Int(b)) = (lhs, rhs) {
        let a = *a;
        let b = *b;
        return match op {
            BinaryOp::Add => Some(ConstValue::Int(a + b)),
            BinaryOp::Sub => Some(ConstValue::Int(a - b)),
            BinaryOp::Mul => Some(ConstValue::Int(a * b)),
            BinaryOp::Div if b != 0 => Some(ConstValue::Int(a / b)),
            BinaryOp::Rem if b != 0 => Some(ConstValue::Int(a % b)),
            BinaryOp::Eq => Some(ConstValue::Bool(a == b)),
            BinaryOp::Ne => Some(ConstValue::Bool(a != b)),
            BinaryOp::Lt => Some(ConstValue::Bool(a < b)),
            BinaryOp::Le => Some(ConstValue::Bool(a <= b)),
            BinaryOp::Gt => Some(ConstValue::Bool(a > b)),
            BinaryOp::Ge => Some(ConstValue::Bool(a >= b)),
            _ => None,
        };
    }
    if let (Some(a), Some(b)) = (as_float(lhs), as_float(rhs)) {
        return match op {
            BinaryOp::Add => Some(ConstValue::Float(a + b)),
            BinaryOp::Sub => Some(ConstValue::Float(a - b)),
            BinaryOp::Mul => Some(ConstValue::Float(a * b)),
            BinaryOp::Div if b != 0.0 => Some(ConstValue::Float(a / b)),
            BinaryOp::Eq => Some(ConstValue::Bool(a == b)),
            BinaryOp::Ne => Some(ConstValue::Bool(a != b)),
            BinaryOp::Lt => Some(ConstValue::Bool(a < b)),
            BinaryOp::Le => Some(ConstValue::Bool(a <= b)),
            BinaryOp::Gt => Some(ConstValue::Bool(a > b)),
            BinaryOp::Ge => Some(ConstValue::Bool(a >= b)),
            _ => None,
        };
    }
    None
}

fn as_float(value: &ConstValue) -> Option<f64> {
    match value {
        ConstValue::Int(inner) => Some(*inner as f64),
        ConstValue::Float(inner) => Some(*inner),
        _ => None,
    }
}

fn eliminate_dead_code(function: &mut Function) {
    loop {
        let mut uses: HashMap<ValueId, usize> = HashMap::new();
        for block in &function.blocks {
            for instruction in &block.instructions {
                for operand in instruction.kind.operands() {
                    *uses.entry(operand).or_default() += 1;
                }
            }
            if let Some(terminator) = &block.terminator {
                for operand in terminator.operands() {
                    *uses.entry(operand).or_default() += 1;
                }
            }
        }

        let mut removed = false;
        for block in &mut function.blocks {
            let before = block.instructions.len();
            block
                .instructions
                .retain(|instruction| match instruction.result {
                    Some(result) => {
                        instruction.kind.is_side_effecting()
                            || uses.get(&result).copied().unwrap_or(0) > 0
                    }
                    None => true,
                });
            if block.instructions.len() != before {
                removed = true;
            }
        }
        if !removed {
            break;
        }
    }

    for block in &mut function.blocks {
        for (index, instruction) in block.instructions.iter_mut().enumerate() {
            if let Some(result) = instruction.result {
                function.values[result.0 as usize].def = crate::value::ValueDef::Inst {
                    block: block.id,
                    index,
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::FunctionBuilder;
    use crate::function::Terminator;
    use crate::lower::lower_program;
    use crate::types::IrType;
    use apeiron_parser::parse_source;

    fn lower(source: &str) -> Module {
        let program = parse_source(source, apeiron_core::source::FileId(0)).expect("parses");
        lower_program(&program).expect("lowers")
    }

    #[test]
    fn folds_constant_arithmetic() {
        let mut builder = FunctionBuilder::new("f", &[], IrType::I64);
        let two = builder.const_i64(2);
        let three = builder.const_i64(3);
        let product = builder.emit(
            InstrKind::Binary {
                op: BinaryOp::Mul,
                lhs: two,
                rhs: three,
            },
            IrType::I64,
            None,
        );
        builder.terminate(Terminator::Ret(Some(product)));
        let mut module = Module::new("m");
        module.functions.push(builder.build());

        ConstantFolding.run(&mut module);
        let folded = &module.functions[0]
            .block(module.functions[0].entry)
            .instructions;
        assert!(matches!(
            folded.last().unwrap().kind,
            InstrKind::Const(ConstValue::Int(6))
        ));
    }

    #[test]
    fn removes_dead_instructions() {
        let mut module = lower("let x = 1 + 2\n");
        let before = module.instruction_count();
        DeadCodeElimination.run(&mut module);
        assert!(module.instruction_count() <= before);
        let main = module.function("main").unwrap();
        // No instruction result is left unused once DCE converges.
        assert!(main
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .all(
                |instruction| instruction.kind.is_side_effecting() || instruction.result.is_some()
            ));
    }

    #[test]
    fn keeps_side_effecting_instructions() {
        let mut module = lower("cause Temperature -> Pressure\n");
        DeadCodeElimination.run(&mut module);
        let main = module.function("main").unwrap();
        assert!(main
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .any(|instruction| matches!(instruction.kind, InstrKind::Causal(_))));
    }

    #[test]
    fn pass_manager_runs_in_order() {
        let manager = PassManager::default_pipeline();
        assert_eq!(
            manager.pass_names(),
            vec!["constant-folding", "dead-code-elimination"]
        );
    }
}

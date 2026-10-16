//! Structural verification of a lowered module.
//!
//! The verifier checks the invariants the rest of the pipeline relies on:
//! every block is terminated, every referenced value is defined and
//! dominates its use, branch arguments match target block parameters, and
//! returns agree with the declared return type.

use std::collections::HashMap;
use std::fmt;

use crate::function::{Function, Module, Terminator};
use crate::types::IrType;
use crate::value::{BlockId, ValueId};

/// A single verification failure.
#[derive(Clone, Debug)]
pub struct VerifyError {
    /// Function the error was found in.
    pub function: String,
    /// What went wrong.
    pub kind: VerifyErrorKind,
}

/// Categories of verification failure.
#[derive(Clone, Debug)]
pub enum VerifyErrorKind {
    /// A block has no terminator.
    MissingTerminator {
        /// Offending block.
        block: BlockId,
    },
    /// A terminator or instruction references an out-of-range block.
    UnknownBlock {
        /// Offending block.
        block: BlockId,
    },
    /// A terminator or instruction references an out-of-range value.
    UnknownValue {
        /// Offending value.
        value: ValueId,
    },
    /// A value is used in a block it does not dominate.
    NotDominated {
        /// Offending value.
        value: ValueId,
        /// Block that uses it.
        block: BlockId,
    },
    /// Branch arguments do not match the target's parameters.
    ArgCountMismatch {
        /// Branching block.
        block: BlockId,
        /// Target block.
        target: BlockId,
        /// Expected argument count.
        expected: usize,
        /// Provided argument count.
        found: usize,
    },
    /// A branch condition is not `bool`.
    ConditionNotBool {
        /// Branching block.
        block: BlockId,
    },
    /// A `ret` value type does not match the function return type.
    ReturnTypeMismatch {
        /// Type found on the value.
        found: IrType,
        /// Declared return type.
        expected: IrType,
    },
    /// A `ret` is missing a value the return type requires.
    MissingReturnValue {
        /// Declared return type.
        expected: IrType,
    },
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "in `{}`: ", self.function)?;
        match &self.kind {
            VerifyErrorKind::MissingTerminator { block } => {
                write!(f, "block {block} has no terminator")
            }
            VerifyErrorKind::UnknownBlock { block } => write!(f, "unknown block {block}"),
            VerifyErrorKind::UnknownValue { value } => write!(f, "unknown value {value}"),
            VerifyErrorKind::NotDominated { value, block } => {
                write!(f, "value {value} does not dominate its use in {block}")
            }
            VerifyErrorKind::ArgCountMismatch {
                block,
                target,
                expected,
                found,
            } => write!(
                f,
                "block {block} passes {found} argument(s) to {target}, which expects {expected}"
            ),
            VerifyErrorKind::ConditionNotBool { block } => {
                write!(f, "branch condition in {block} is not bool")
            }
            VerifyErrorKind::ReturnTypeMismatch { found, expected } => {
                write!(f, "return type {found} does not match declared {expected}")
            }
            VerifyErrorKind::MissingReturnValue { expected } => {
                write!(f, "return is missing a value of type {expected}")
            }
        }
    }
}

impl std::error::Error for VerifyError {}

/// Verifies every function in the module.
pub fn verify_module(module: &Module) -> Result<(), Vec<VerifyError>> {
    let mut errors = Vec::new();
    for function in &module.functions {
        verify_function(function, &mut errors);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verifies a single function, pushing failures into `errors`.
pub fn verify_function(function: &Function, errors: &mut Vec<VerifyError>) {
    let dominators = compute_dominators(function);
    let value_count = function.values.len() as u32;
    let block_count = function.blocks.len() as u32;

    for block in &function.blocks {
        let Some(terminator) = &block.terminator else {
            errors.push(err(
                function,
                VerifyErrorKind::MissingTerminator { block: block.id },
            ));
            continue;
        };

        for instruction in &block.instructions {
            for operand in instruction.kind.operands() {
                check_value(
                    function,
                    operand,
                    block.id,
                    value_count,
                    &dominators,
                    errors,
                );
            }
        }

        for operand in terminator.operands() {
            check_value(
                function,
                operand,
                block.id,
                value_count,
                &dominators,
                errors,
            );
        }

        match terminator {
            Terminator::Jump { target, args } => {
                check_target(function, block.id, *target, args.len(), block_count, errors);
            }
            Terminator::Branch {
                condition,
                then_block,
                then_args,
                else_block,
                else_args,
            } => {
                if let Some(ty) = value_type(function, *condition) {
                    if *ty != IrType::Bool {
                        errors.push(err(
                            function,
                            VerifyErrorKind::ConditionNotBool { block: block.id },
                        ));
                    }
                }
                check_target(
                    function,
                    block.id,
                    *then_block,
                    then_args.len(),
                    block_count,
                    errors,
                );
                check_target(
                    function,
                    block.id,
                    *else_block,
                    else_args.len(),
                    block_count,
                    errors,
                );
            }
            Terminator::Ret(value) => match value {
                Some(value) => {
                    if let Some(found) = value_type(function, *value) {
                        if found != &function.return_type {
                            errors.push(err(
                                function,
                                VerifyErrorKind::ReturnTypeMismatch {
                                    found: found.clone(),
                                    expected: function.return_type.clone(),
                                },
                            ));
                        }
                    }
                }
                None => {
                    if function.return_type != IrType::Unit {
                        errors.push(err(
                            function,
                            VerifyErrorKind::MissingReturnValue {
                                expected: function.return_type.clone(),
                            },
                        ));
                    }
                }
            },
            Terminator::Unreachable => {}
        }
    }
}

fn err(function: &Function, kind: VerifyErrorKind) -> VerifyError {
    VerifyError {
        function: function.name.clone(),
        kind,
    }
}

fn value_type(function: &Function, value: ValueId) -> Option<&IrType> {
    function.values.get(value.0 as usize).map(|value| &value.ty)
}

fn check_value(
    function: &Function,
    value: ValueId,
    block: BlockId,
    value_count: u32,
    dominators: &HashMap<BlockId, Vec<bool>>,
    errors: &mut Vec<VerifyError>,
) {
    if value.0 >= value_count {
        errors.push(err(function, VerifyErrorKind::UnknownValue { value }));
        return;
    }
    let def_block = function.value(value).block();
    let dominated = dominators
        .get(&block)
        .and_then(|set| set.get(def_block.0 as usize).copied())
        .unwrap_or(false);
    if !dominated {
        errors.push(err(
            function,
            VerifyErrorKind::NotDominated { value, block },
        ));
    }
}

fn check_target(
    function: &Function,
    block: BlockId,
    target: BlockId,
    arg_count: usize,
    block_count: u32,
    errors: &mut Vec<VerifyError>,
) {
    if target.0 >= block_count {
        errors.push(err(
            function,
            VerifyErrorKind::UnknownBlock { block: target },
        ));
        return;
    }
    let expected = function.block(target).params.len();
    if expected != arg_count {
        errors.push(err(
            function,
            VerifyErrorKind::ArgCountMismatch {
                block,
                target,
                expected,
                found: arg_count,
            },
        ));
    }
}

/// Iterative dominator computation over the CFG.
fn compute_dominators(function: &Function) -> HashMap<BlockId, Vec<bool>> {
    let count = function.blocks.len();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); count];
    for block in &function.blocks {
        if let Some(terminator) = &block.terminator {
            for successor in terminator.successors() {
                let target = successor.0 as usize;
                if target < count {
                    preds[target].push(block.id.0 as usize);
                }
            }
        }
    }

    let entry = function.entry.0 as usize;
    let mut dom: Vec<Vec<bool>> = vec![vec![true; count]; count];
    dom[entry] = vec![false; count];
    dom[entry][entry] = true;

    let mut changed = true;
    while changed {
        changed = false;
        for block in 0..count {
            if block == entry {
                continue;
            }
            let mut new_dom = vec![true; count];
            if preds[block].is_empty() {
                new_dom = vec![false; count];
                new_dom[block] = true;
            } else {
                for predecessor in &preds[block] {
                    for other in 0..count {
                        new_dom[other] &= dom[*predecessor][other];
                    }
                }
                new_dom[block] = true;
            }
            if new_dom != dom[block] {
                dom[block] = new_dom;
                changed = true;
            }
        }
    }

    let mut result = HashMap::new();
    for block in &function.blocks {
        result.insert(block.id, dom[block.id.0 as usize].clone());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::FunctionBuilder;
    use crate::instr::InstrKind;
    use crate::value::ValueDef;

    #[test]
    fn verifies_a_well_formed_function() {
        let mut builder = FunctionBuilder::new("f", &[IrType::I64], IrType::I64);
        let param = builder.param(0);
        let one = builder.const_i64(1);
        let sum = builder.emit(
            InstrKind::Binary {
                op: apeiron_ast::BinaryOp::Add,
                lhs: param,
                rhs: one,
            },
            IrType::I64,
            None,
        );
        builder.terminate(Terminator::Ret(Some(sum)));
        let function = builder.build();
        let mut errors = Vec::new();
        verify_function(&function, &mut errors);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn detects_missing_terminator() {
        let builder = FunctionBuilder::new("f", &[], IrType::Unit);
        let function = builder.build();
        let mut errors = Vec::new();
        verify_function(&function, &mut errors);
        assert!(matches!(
            errors[0].kind,
            VerifyErrorKind::MissingTerminator { .. }
        ));
    }

    #[test]
    fn detects_return_type_mismatch() {
        let mut builder = FunctionBuilder::new("f", &[], IrType::F32);
        let value = builder.const_i64(0);
        builder.terminate(Terminator::Ret(Some(value)));
        let function = builder.build();
        let mut errors = Vec::new();
        verify_function(&function, &mut errors);
        assert!(errors
            .iter()
            .any(|error| matches!(error.kind, VerifyErrorKind::ReturnTypeMismatch { .. })));
    }

    #[test]
    fn detects_unknown_value() {
        let mut builder = FunctionBuilder::new("f", &[], IrType::Unit);
        let bogus = ValueId(999);
        builder.emit_effect(
            InstrKind::Call {
                callee: "g".into(),
                args: vec![bogus],
            },
            None,
        );
        builder.terminate(Terminator::Ret(None));
        let function = builder.build();
        let mut errors = Vec::new();
        verify_function(&function, &mut errors);
        assert!(errors
            .iter()
            .any(|error| matches!(error.kind, VerifyErrorKind::UnknownValue { .. })));
    }

    #[test]
    fn detects_missing_arg_count() {
        let mut builder = FunctionBuilder::new("f", &[], IrType::Unit);
        let merge = builder.create_block_with_params("merge", &[IrType::I64]);
        let zero = builder.const_i64(0);
        builder.terminate(Terminator::Jump {
            target: merge,
            args: Vec::new(),
        });
        builder.switch_to(merge);
        builder.terminate(Terminator::Ret(None));
        let _ = zero;
        let function = builder.build();
        let mut errors = Vec::new();
        verify_function(&function, &mut errors);
        assert!(errors
            .iter()
            .any(|error| matches!(error.kind, VerifyErrorKind::ArgCountMismatch { .. })));
    }

    #[test]
    fn value_def_reports_block() {
        let builder = FunctionBuilder::new("f", &[IrType::F32], IrType::F32);
        let param = builder.param(0);
        assert_eq!(
            builder.function().value(param).def,
            ValueDef::Param {
                block: builder.entry(),
                index: 0
            }
        );
    }
}

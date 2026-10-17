//! Human readable rendering of a module, useful for `--emit=ir` and tests.

use std::fmt::Write as _;

use crate::function::{Function, GlobalInit, Module, Terminator};
use crate::instr::{InstrKind, Instruction};
use crate::value::ValueId;

/// Renders a whole module.
pub fn print_module(module: &Module) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "module {}", module.name);
    if !module.globals.is_empty() {
        out.push('\n');
        for global in &module.globals {
            let _ = write!(out, "  global {} : {}", global.name, global.ty);
            if let Some(init) = &global.init {
                match init {
                    GlobalInit::Const(value) => {
                        let _ = write!(out, " = {value}");
                    }
                    GlobalInit::Tensor(literal) => {
                        let _ = write!(out, " = tensor<{}>", literal.elem);
                    }
                }
            }
            out.push('\n');
        }
    }
    for function in &module.functions {
        out.push('\n');
        out.push_str(&print_function(function));
    }
    out
}

/// Renders a single function.
pub fn print_function(function: &Function) -> String {
    let mut out = String::new();
    let params = function
        .params
        .iter()
        .map(|value| format!("{value}: {}", function.value_type(*value)))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(
        out,
        "fn {}({params}) -> {} {{",
        function.name, function.return_type
    );

    for block in &function.blocks {
        let block_params = block
            .params
            .iter()
            .map(|value| format!("{value}: {}", function.value_type(*value)))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "  {} \"{}\"({block_params}):", block.id, block.label);
        for instruction in &block.instructions {
            let _ = writeln!(out, "    {}", format_instruction(instruction));
        }
        if let Some(terminator) = &block.terminator {
            let _ = writeln!(out, "    {}", format_terminator(terminator));
        }
    }
    out.push_str("}\n");
    out
}

fn format_instruction(instruction: &Instruction) -> String {
    let body = format_kind(&instruction.kind);
    match instruction.result {
        Some(result) => format!("{result} = {body}"),
        None => body,
    }
}

fn format_kind(kind: &InstrKind) -> String {
    match kind {
        InstrKind::Const(value) => format!("const {value}"),
        InstrKind::TensorLit(literal) => format!("tensor.lit {} <{}>", literal.elem, literal.len()),
        InstrKind::GlobalRef { name } => format!("global {name:?}"),
        InstrKind::MakeTensor { elements } => {
            format!("make_tensor [{}]", join(elements))
        }
        InstrKind::Unary { op, operand } => format!("unary {op} {operand}"),
        InstrKind::Binary { op, lhs, rhs } => format!("binary {op} {lhs}, {rhs}"),
        InstrKind::MatMul { lhs, rhs } => format!("matmul {lhs}, {rhs}"),
        InstrKind::Call { callee, args } => {
            format!("call {callee}({})", join(args))
        }
        InstrKind::Index { target, index } => format!("index {target}[{index}]"),
        InstrKind::Extract { tensor, index } => format!("extract {tensor}[{index}]"),
        InstrKind::Dimension { tensor, axis } => format!("dim {tensor}, {axis}"),
        InstrKind::MakeEnum {
            type_name,
            variant,
            payload,
        } => format!("make_enum {type_name}::{variant}({})", join(payload)),
        InstrKind::EnumIs { value, variant } => format!("enum_is {value}, {variant}"),
        InstrKind::EnumPayload { value, index } => format!("enum_payload {value}[{index}]"),
        InstrKind::Gradient { target, wrt } => format!("gradient {target} wrt {wrt}"),
        InstrKind::Belief(inner) => format!("{inner:?}"),
        InstrKind::Distribution { name, categories } => {
            format!("distribution {name:?} ({} categories)", categories.len())
        }
        InstrKind::Causal(inner) => format!("{inner:?}"),
        InstrKind::Memory(inner) => format!("{inner:?}"),
        InstrKind::Architecture(inner) => format!("{inner:?}"),
        InstrKind::Society(inner) => format!("{inner:?}"),
        InstrKind::Plan { name, goal } => format!("plan {name:?} goal {goal}"),
        InstrKind::Optimizer { kind, settings } => {
            format!("optimizer {kind} ({} settings)", settings.len())
        }
        InstrKind::Train { model, dataset } => format!("train {model} on {dataset}"),
        InstrKind::Route {
            request,
            cluster,
            budget,
        } => {
            format!(
                "route {request} to {cluster:?} ({} budget entries)",
                budget.len()
            )
        }
    }
}

fn format_terminator(terminator: &Terminator) -> String {
    match terminator {
        Terminator::Ret(None) => "ret".to_string(),
        Terminator::Ret(Some(value)) => format!("ret {value}"),
        Terminator::Jump { target, args } => format!("br {target}({})", join(args)),
        Terminator::Branch {
            condition,
            then_block,
            then_args,
            else_block,
            else_args,
        } => format!(
            "condbr {condition}, {then_block}({}), {else_block}({})",
            join(then_args),
            join(else_args)
        ),
        Terminator::Unreachable => "unreachable".to_string(),
    }
}

fn join(values: &[ValueId]) -> String {
    values
        .iter()
        .map(ValueId::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::FunctionBuilder;
    use crate::lower::lower_program;
    use crate::types::IrType;
    use apeiron_parser::parse_source;

    #[test]
    fn prints_a_function() {
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
        let text = print_function(&builder.build());
        assert!(text.contains("fn f(%0: i64) -> i64 {"));
        assert!(text.contains("binary + %0, %1"));
        assert!(text.contains("ret %2"));
    }

    #[test]
    fn prints_a_module_with_globals() {
        let program = parse_source(
            "tensor W : f32[2] = [1.0, 2.0]\n",
            apeiron_core::source::FileId(0),
        )
        .expect("parses");
        let module = lower_program(&program).expect("lowers");
        let text = print_module(&module);
        assert!(text.contains("module apeiron"));
        assert!(text.contains("global W : f32[2]"));
        assert!(text.contains("fn main()"));
    }
}

//! Canonical source-like pretty printer.
//!
//! Rendering the AST back to text is used by tests and by tooling that
//! wants to show normalized code. The output is deterministic and
//! precedence aware, but not byte-for-byte identical to the input
//! (comments and original whitespace are not preserved by the AST).

use std::fmt;

use crate::expr::Expr;
use crate::ident::Ident;
use crate::item::{
    ArchOp, BeliefStmt, Field, Item, MemoryDecl, Program, SocietyBlock, SocietyMember,
};

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (idx, item) in self.items.iter().enumerate() {
            if idx > 0 {
                f.write_str("\n")?;
            }
            write!(f, "{item}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_expr(f, self, 0)
    }
}

fn fmt_expr(f: &mut fmt::Formatter<'_>, expr: &Expr, parent: u8) -> fmt::Result {
    let precedence = expr.precedence();
    let wrap = precedence < parent;
    if wrap {
        f.write_str("(")?;
    }
    match expr {
        Expr::Literal { value, .. } => write!(f, "{value}")?,
        Expr::Ident(ident) => write!(f, "{ident}")?,
        Expr::Array { elements, .. } => {
            f.write_str("[")?;
            for (idx, element) in elements.iter().enumerate() {
                if idx > 0 {
                    f.write_str(", ")?;
                }
                fmt_expr(f, element, 0)?;
            }
            f.write_str("]")?;
        }
        Expr::Unary { op, operand, .. } => {
            write!(f, "{op}")?;
            fmt_expr(f, operand, op.precedence() + 1)?;
        }
        Expr::Binary { op, lhs, rhs, .. } => {
            fmt_expr(f, lhs, precedence)?;
            write!(f, " {op} ")?;
            fmt_expr(f, rhs, precedence + 1)?;
        }
        Expr::Call { callee, args, .. } => {
            fmt_expr(f, callee, 8)?;
            f.write_str("(")?;
            for (idx, arg) in args.iter().enumerate() {
                if idx > 0 {
                    f.write_str(", ")?;
                }
                fmt_expr(f, arg, 0)?;
            }
            f.write_str(")")?;
        }
        Expr::Index { target, index, .. } => {
            fmt_expr(f, target, 8)?;
            f.write_str("[")?;
            fmt_expr(f, index, 0)?;
            f.write_str("]")?;
        }
        Expr::Field { target, field, .. } => {
            fmt_expr(f, target, 8)?;
            write!(f, ".{field}")?;
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            f.write_str("if ")?;
            fmt_expr(f, condition, 0)?;
            f.write_str(" ")?;
            fmt_expr(f, then_branch, 0)?;
            if let Some(branch) = else_branch {
                f.write_str(" else ")?;
                fmt_expr(f, branch, 0)?;
            }
        }
        Expr::Block { items, .. } => fmt_block(f, items)?,
    }
    if wrap {
        f.write_str(")")?;
    }
    Ok(())
}

fn fmt_block(f: &mut fmt::Formatter<'_>, items: &[Item]) -> fmt::Result {
    f.write_str("{")?;
    if !items.is_empty() {
        f.write_str(" ")?;
        for (idx, item) in items.iter().enumerate() {
            if idx > 0 {
                f.write_str("; ")?;
            }
            write!(f, "{item}")?;
        }
        f.write_str(" ")?;
    }
    f.write_str("}")
}

fn fmt_type_annotation(ty: &Option<crate::types::TypeExpr>) -> String {
    match ty {
        Some(ty) => format!(" : {ty}"),
        None => String::new(),
    }
}

fn fmt_colon_fields(f: &mut fmt::Formatter<'_>, fields: &[Field]) -> fmt::Result {
    f.write_str("{ ")?;
    for (idx, field) in fields.iter().enumerate() {
        if idx > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{}: {}", field.name, field.value)?;
    }
    f.write_str(" }")
}

fn fmt_eq_fields(f: &mut fmt::Formatter<'_>, fields: &[Field]) -> fmt::Result {
    for field in fields {
        write!(f, " {} = {}", field.name, field.value)?;
    }
    Ok(())
}

fn fmt_ident_list(f: &mut fmt::Formatter<'_>, idents: &[Ident]) -> fmt::Result {
    f.write_str("[")?;
    for (idx, ident) in idents.iter().enumerate() {
        if idx > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{ident}")?;
    }
    f.write_str("]")
}

impl fmt::Display for ArchOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArchOp::AddComponent { component, .. } => write!(f, "add component {component}"),
            ArchOp::ExpandMemory { memory, by, .. } => {
                write!(f, "expand memory {memory} by {by}")
            }
            ArchOp::ChangeRouting { routing, .. } => write!(f, "change routing {routing}"),
        }
    }
}

impl fmt::Display for SocietyMember {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SocietyMember::Agent { name, fields, .. } => {
                write!(f, "agent {name} ")?;
                fmt_colon_fields(f, fields)
            }
            SocietyMember::Cluster { name, members, .. } => {
                write!(f, "cluster {name} = ")?;
                fmt_ident_list(f, members)
            }
        }
    }
}

impl fmt::Display for SocietyBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "society {} {{ ", self.name)?;
        for (idx, member) in self.members.iter().enumerate() {
            if idx > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{member}")?;
        }
        f.write_str(" }")
    }
}

impl fmt::Display for MemoryDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "memory {} {} ", self.tier, self.name)?;
        fmt_colon_fields(f, &self.fields)
    }
}

impl fmt::Display for BeliefStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BeliefStmt::Declare { name, fields, .. } => {
                write!(f, "belief {name} ")?;
                fmt_colon_fields(f, fields)
            }
            BeliefStmt::Update { name, fields, .. } => {
                write!(f, "belief {name} update ")?;
                fmt_colon_fields(f, fields)
            }
        }
    }
}

impl fmt::Display for Item {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Item::Fn(decl) => {
                write!(f, "fn {}(", decl.name)?;
                for (idx, param) in decl.params.iter().enumerate() {
                    if idx > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", param.name)?;
                    if let Some(ty) = &param.ty {
                        write!(f, ": {ty}")?;
                    }
                }
                f.write_str(")")?;
                if let Some(ret) = &decl.return_type {
                    write!(f, " -> {ret}")?;
                }
                write!(f, " {}", decl.body)
            }
            Item::Let(decl) => {
                write!(f, "let {}{}", decl.name, fmt_type_annotation(&decl.ty))?;
                write!(f, " = {}", decl.value)
            }
            Item::Tensor(decl) => {
                write!(
                    f,
                    "{} {}{}",
                    decl.kind,
                    decl.name,
                    fmt_type_annotation(&decl.ty)
                )?;
                write!(f, " = {}", decl.value)
            }
            Item::Return { value, .. } => match value {
                Some(value) => write!(f, "return {value}"),
                None => f.write_str("return"),
            },
            Item::While {
                condition, body, ..
            } => write!(f, "while {condition} {body}"),
            Item::For {
                variable,
                iterable,
                body,
                ..
            } => {
                write!(f, "for {variable} in {iterable} {body}")
            }
            Item::Break { .. } => f.write_str("break"),
            Item::Continue { .. } => f.write_str("continue"),
            Item::Expr(expr) => write!(f, "{expr}"),
            Item::Optimizer { kind, settings, .. } => {
                write!(f, "optimizer {kind} ")?;
                fmt_colon_fields(f, settings)
            }
            Item::Train { model, dataset, .. } => write!(f, "train {model} on {dataset}"),
            Item::Gradient {
                target,
                with_respect_to,
                ..
            } => {
                write!(f, "gradient {target} with_respect_to {with_respect_to}")
            }
            Item::Distribution {
                name, categories, ..
            } => {
                write!(f, "distribution {name} ")?;
                fmt_colon_fields(f, categories)
            }
            Item::Belief(stmt) => write!(f, "{stmt}"),
            Item::Cause { from, to, .. } => write!(f, "cause {from} -> {to}"),
            Item::Observe { target, value, .. } => write!(f, "observe {target} = {value}"),
            Item::Intervene { target, value, .. } => write!(f, "intervene {target} = {value}"),
            Item::Counterfactual {
                target, condition, ..
            } => {
                write!(f, "counterfactual {target} if {condition}")
            }
            Item::Simulate {
                parameter, value, ..
            } => write!(f, "simulate {parameter} = {value}"),
            Item::Memory(decl) => write!(f, "{decl}"),
            Item::Remember {
                memory, entries, ..
            } => {
                write!(f, "remember {memory}")?;
                fmt_eq_fields(f, entries)
            }
            Item::Consolidate {
                from,
                to,
                threshold,
                ..
            } => {
                write!(f, "consolidate {from} -> {to}")?;
                match threshold {
                    Some(value) => write!(f, " threshold = {value}"),
                    None => Ok(()),
                }
            }
            Item::Recall {
                memory, arguments, ..
            } => {
                write!(f, "recall {memory}")?;
                fmt_eq_fields(f, arguments)
            }
            Item::Plan { name, body, .. } => {
                write!(f, "plan {name} ")?;
                fmt_colon_fields(f, body)
            }
            Item::SelfModel { name, body, .. } => {
                write!(f, "self_model {name} ")?;
                fmt_colon_fields(f, body)
            }
            Item::Architecture(block) => {
                write!(f, "architecture {} {{ ", block.name)?;
                for (idx, op) in block.operations.iter().enumerate() {
                    if idx > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{op}")?;
                }
                f.write_str(" }")
            }
            Item::Society(block) => write!(f, "{block}"),
            Item::Route {
                request,
                cluster,
                budget,
                ..
            } => {
                write!(f, "route {request} to {cluster} budget ")?;
                fmt_colon_fields(f, budget)
            }
        }
    }
}

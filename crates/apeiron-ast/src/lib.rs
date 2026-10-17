//! Abstract syntax tree for Apeiron.
//!
//! The tree is deliberately shallow: every construct that the language
//! treats as a first-class primitive — tensor declarations, epistemic
//! beliefs, causal edges, memory operations, architectural mutations,
//! and agent societies — has a dedicated node instead of being lowered
//! to generic function calls.
//!
//! Nodes store [`Span`](apeiron_core::span::Span)s so that later stages
//! can emit precise diagnostics without re-reading the source.

pub mod builtins;
pub mod expr;
pub mod ident;
pub mod item;
pub mod print;
pub mod types;

pub use builtins::{BuiltinRet, BuiltinSig, BUILTINS};
pub use expr::{Arm, BinaryOp, Expr, Literal, Pattern, UnaryOp};
pub use ident::Ident;
pub use item::{
    ArchOp, ArchitectureBlock, BeliefStmt, Field, FnDecl, Item, LetDecl, MemoryDecl, MemoryTier,
    Param, Program, SocietyBlock, SocietyMember, TensorDecl, TensorKind,
};
pub use types::{TypeDim, TypeExpr};

#[cfg(test)]
mod tests {
    use super::*;
    use apeiron_core::source::FileId;
    use apeiron_core::span::Span;

    fn span() -> Span {
        Span::empty(FileId(0))
    }

    fn ident(name: &str) -> Ident {
        Ident::new(name, span())
    }

    fn int(value: i64) -> Expr {
        Expr::Literal {
            value: Literal::Int(value),
            span: span(),
        }
    }

    fn float(value: f64) -> Expr {
        Expr::Literal {
            value: Literal::Float(value),
            span: span(),
        }
    }

    #[test]
    fn binary_expr_span_covers_operands() {
        let lhs = Expr::Ident(Ident::new(
            "a",
            Span::new(
                apeiron_core::span::Location::new(FileId(0), 0, 1, 1),
                apeiron_core::span::Location::new(FileId(0), 1, 1, 2),
            ),
        ));
        let rhs = Expr::Ident(Ident::new(
            "b",
            Span::new(
                apeiron_core::span::Location::new(FileId(0), 4, 1, 5),
                apeiron_core::span::Location::new(FileId(0), 5, 1, 6),
            ),
        ));
        let span = lhs.span().merge(rhs.span());
        let binary = Expr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            span,
        };
        assert_eq!(binary.span().start.column, 1);
        assert_eq!(binary.span().end.column, 6);
    }

    #[test]
    fn precedence_orders_operators() {
        assert!(BinaryOp::MatMul.precedence() > BinaryOp::Mul.precedence());
        assert!(BinaryOp::Mul.precedence() > BinaryOp::Add.precedence());
        assert!(BinaryOp::Add.precedence() > BinaryOp::Eq.precedence());
        assert!(BinaryOp::Eq.precedence() > BinaryOp::Assign.precedence());
        assert!(BinaryOp::Le.is_comparison());
        assert!(!BinaryOp::Add.is_comparison());
    }

    #[test]
    fn printer_keeps_precedence() {
        let product = Expr::Binary {
            op: BinaryOp::Mul,
            lhs: Box::new(int(2)),
            rhs: Box::new(int(3)),
            span: span(),
        };
        let sum = Expr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(int(1)),
            rhs: Box::new(product.clone()),
            span: span(),
        };
        assert_eq!(sum.to_string(), "1 + 2 * 3");

        let grouped = Expr::Binary {
            op: BinaryOp::Mul,
            lhs: Box::new(sum),
            rhs: Box::new(int(4)),
            span: span(),
        };
        assert_eq!(grouped.to_string(), "(1 + 2 * 3) * 4");
    }

    #[test]
    fn printer_handles_postfix_and_unary() {
        let call = Expr::Call {
            callee: Box::new(Expr::Ident(ident("recall"))),
            args: vec![Expr::Ident(ident("Mem")), int(3)],
            span: span(),
        };
        assert_eq!(call.to_string(), "recall(Mem, 3)");

        let field = Expr::Field {
            target: Box::new(Expr::Ident(ident("society"))),
            field: ident("Planner"),
            span: span(),
        };
        assert_eq!(field.to_string(), "society.Planner");

        let negated = Expr::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(float(0.5)),
            span: span(),
        };
        assert_eq!(negated.to_string(), "-0.5");

        let nested = Expr::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(negated),
            span: span(),
        };
        assert_eq!(nested.to_string(), "-(-0.5)");
    }

    #[test]
    fn printer_renders_domain_items() {
        let item = Item::Cause {
            from: ident("Temperature"),
            to: ident("Pressure"),
            span: span(),
        };
        assert_eq!(item.to_string(), "cause Temperature -> Pressure");

        let belief = BeliefStmt::Declare {
            name: ident("TargetThreat"),
            fields: vec![Field::new(ident("confidence"), float(0.42), span())],
            span: span(),
        };
        assert_eq!(
            Item::Belief(belief.clone()).to_string(),
            "belief TargetThreat { confidence: 0.42 }"
        );
        assert!(!belief.is_update());
        assert_eq!(belief.name().name, "TargetThreat");
        assert_eq!(belief.fields().len(), 1);

        let memory = MemoryDecl {
            tier: MemoryTier::Episodic,
            name: ident("ShortTermMem"),
            fields: vec![Field::new(ident("dimension"), int(128), span())],
            span: span(),
        };
        assert_eq!(
            Item::Memory(memory).to_string(),
            "memory episodic ShortTermMem { dimension: 128 }"
        );
    }

    #[test]
    fn arch_op_display() {
        let op = ArchOp::ExpandMemory {
            memory: Expr::Literal {
                value: Literal::Str("ShortTermMem".into()),
                span: span(),
            },
            by: int(2000),
            span: span(),
        };
        assert_eq!(op.to_string(), "expand memory \"ShortTermMem\" by 2000");
        assert_eq!(op.span(), span());
    }

    #[test]
    fn society_member_display() {
        let cluster = SocietyMember::Cluster {
            name: ident("MainCluster"),
            members: vec![ident("Planner"), ident("Executor")],
            span: span(),
        };
        assert_eq!(
            cluster.to_string(),
            "cluster MainCluster = [Planner, Executor]"
        );
        assert_eq!(cluster.name().name, "MainCluster");
    }

    #[test]
    fn item_span_delegates_to_node() {
        let span = Span::empty(FileId(3));
        let item = Item::Break { span };
        assert_eq!(item.span(), span);
    }
}

//! Expression nodes of the Apeiron AST.

use std::fmt;

use apeiron_core::span::Span;

use crate::ident::Ident;
use crate::item::Item;

/// Literal values understood by the language.
#[derive(Clone, PartialEq, Debug)]
pub enum Literal {
    /// Integer literal such as `50`.
    Int(i64),
    /// Floating point literal such as `0.001`.
    Float(f64),
    /// String literal with escapes already decoded.
    Str(String),
    /// `true` or `false`.
    Bool(bool),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Int(value) => write!(f, "{value}"),
            Literal::Float(value) => write!(f, "{value:?}"),
            Literal::Str(value) => write!(f, "{value:?}"),
            Literal::Bool(value) => write!(f, "{value}"),
        }
    }
}

/// Pattern used in `match` arms.
#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    /// Match any value and bind it to a name.
    Bind(Ident),
    /// Match a literal value.
    Literal(Literal),
    /// Match and ignore.
    Wildcard,
    /// Match a tuple/list of patterns.
    Tuple(Vec<Pattern>),
    /// Match an enum variant: `Some(x)`, `None`, `Ok(v)`, `Err(e)`.
    Variant {
        /// The variant name.
        name: Ident,
        /// The payload pattern, when the variant carries a value.
        inner: Option<Box<Pattern>>,
    },
}

impl Pattern {
    /// Whether the pattern matches every value of any type without a guard.
    ///
    /// Tuple and variant patterns are always refutable: arity and the
    /// variant tag must still be checked at match time.
    pub fn is_irrefutable(&self) -> bool {
        matches!(self, Pattern::Wildcard | Pattern::Bind(_))
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pattern::Bind(ident) => ident.fmt(f),
            Pattern::Literal(lit) => lit.fmt(f),
            Pattern::Wildcard => f.write_str("_"),
            Pattern::Tuple(pats) => {
                f.write_str("(")?;
                for (index, pat) in pats.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    pat.fmt(f)?;
                }
                f.write_str(")")
            }
            Pattern::Variant { name, inner } => {
                write!(f, "{name}")?;
                if let Some(inner) = inner {
                    write!(f, "({inner})")?;
                }
                Ok(())
            }
        }
    }
}

/// A match arm `pattern [if guard] => body`.
#[derive(Clone, Debug, PartialEq)]
pub struct Arm {
    /// The pattern to match against.
    pub pat: Pattern,
    /// Optional guard condition.
    pub guard: Option<Expr>,
    /// The body expression evaluated on match.
    pub body: Expr,
    /// Source range of the arm.
    pub span: Span,
}

impl Arm {
    /// Creates a new arm.
    pub fn new(pat: Pattern, guard: Option<Expr>, body: Expr, span: Span) -> Self {
        Self {
            pat,
            guard,
            body,
            span,
        }
    }
}

impl fmt::Display for Arm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} => {}", self.pat, self.body)
    }
}

/// Prefix operators.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnaryOp {
    /// Arithmetic negation: `-x`.
    Neg,
    /// Logical negation: `!x`.
    Not,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => f.write_str("-"),
            UnaryOp::Not => f.write_str("!"),
        }
    }
}

impl UnaryOp {
    /// Binding power of the operator prefix.
    pub const fn precedence(self) -> u8 {
        6
    }
}

/// Infix operators ordered by increasing binding power.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinaryOp {
    /// `=` assignment.
    Assign,
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `%`
    Rem,
    /// `@` matrix multiplication.
    MatMul,
}

impl BinaryOp {
    /// Whether the operator produces a boolean value.
    pub const fn is_comparison(self) -> bool {
        matches!(
            self,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
        )
    }

    /// Binding power used by both the parser and the pretty printer.
    pub const fn precedence(self) -> u8 {
        match self {
            BinaryOp::Assign => 1,
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge => 2,
            BinaryOp::Add | BinaryOp::Sub => 3,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => 4,
            BinaryOp::MatMul => 5,
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            BinaryOp::Assign => "=",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Rem => "%",
            BinaryOp::MatMul => "@",
        };
        f.write_str(text)
    }
}

/// Every expression form in Apeiron.
#[derive(Clone, PartialEq, Debug)]
pub enum Expr {
    /// Literal value.
    Literal {
        /// The literal itself.
        value: Literal,
        /// Source range of the literal.
        span: Span,
    },
    /// Variable or component reference.
    Ident(Ident),
    /// Tensor literal: `[[1.0, 2.0], [3.0, 4.0]]`.
    Array {
        /// Row-major elements; nesting encodes rank.
        elements: Vec<Expr>,
        /// Source range of the whole literal.
        span: Span,
    },
    /// Prefix operator application.
    Unary {
        /// The operator.
        op: UnaryOp,
        /// Its operand.
        operand: Box<Expr>,
        /// Source range.
        span: Span,
    },
    /// Infix operator application.
    Binary {
        /// The operator.
        op: BinaryOp,
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
        /// Source range.
        span: Span,
    },
    /// Function or primitive invocation: `recall(M, top_k)`.
    Call {
        /// Callee expression.
        callee: Box<Expr>,
        /// Positional arguments.
        args: Vec<Expr>,
        /// Source range.
        span: Span,
    },
    /// Indexing: `W[0]`.
    Index {
        /// Indexed expression.
        target: Box<Expr>,
        /// Index expression.
        index: Box<Expr>,
        /// Source range.
        span: Span,
    },
    /// Field access: `society.Planner`.
    Field {
        /// Receiver expression.
        target: Box<Expr>,
        /// Field name.
        field: Ident,
        /// Source range.
        span: Span,
    },
    /// Conditional expression.
    If {
        /// Condition.
        condition: Box<Expr>,
        /// Branch taken when the condition holds.
        then_branch: Box<Expr>,
        /// Optional alternative branch.
        else_branch: Option<Box<Expr>>,
        /// Source range.
        span: Span,
    },
    /// Pattern match expression: `match x { 1 => "one", _ => "other" }`.
    Match {
        /// Scrutinized expression.
        expr: Box<Expr>,
        /// Match arms.
        arms: Vec<Arm>,
        /// Source range.
        span: Span,
    },
    /// Error propagation: `value?`.
    Try {
        /// The `Option`/`Result` expression whose payload is unwrapped.
        operand: Box<Expr>,
        /// Source range, covering the operand and the `?`.
        span: Span,
    },
    /// Brace-delimited sequence of items evaluated as an expression.
    Block {
        /// Items in source order.
        items: Vec<Item>,
        /// Source range.
        span: Span,
    },
}

impl Expr {
    /// Source range covered by this expression.
    pub fn span(&self) -> Span {
        match self {
            Expr::Literal { span, .. }
            | Expr::Array { span, .. }
            | Expr::Unary { span, .. }
            | Expr::Binary { span, .. }
            | Expr::Call { span, .. }
            | Expr::Index { span, .. }
            | Expr::Field { span, .. }
            | Expr::If { span, .. }
            | Expr::Block { span, .. }
            | Expr::Match { span, .. }
            | Expr::Try { span, .. } => *span,
            Expr::Ident(ident) => ident.span,
        }
    }

    /// Whether this expression is a literal.
    pub fn is_literal(&self) -> bool {
        matches!(self, Expr::Literal { .. })
    }

    /// If this expression is a plain identifier, returns it.
    pub fn as_ident(&self) -> Option<&Ident> {
        match self {
            Expr::Ident(ident) => Some(ident),
            _ => None,
        }
    }

    /// If this expression is a string literal, returns its contents.
    pub fn as_str_literal(&self) -> Option<&str> {
        match self {
            Expr::Literal {
                value: Literal::Str(text),
                ..
            } => Some(text),
            _ => None,
        }
    }

    /// Binding power used by the pretty printer.
    pub(crate) fn precedence(&self) -> u8 {
        match self {
            Expr::Binary { op, .. } => op.precedence(),
            Expr::Unary { .. } => UnaryOp::Neg.precedence(),
            Expr::Call { .. } | Expr::Index { .. } | Expr::Field { .. } => 8,
            Expr::Match { .. } => 8,
            Expr::Try { .. } => 8,
            _ => 9,
        }
    }
}

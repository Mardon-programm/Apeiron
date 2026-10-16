//! Statements, declarations, and cognitive primitives of the AST.

use std::fmt;

use apeiron_core::span::Span;

use crate::expr::Expr;
use crate::ident::Ident;
use crate::types::TypeExpr;

/// A `name: value` or `name = value` entry inside a declaration body.
#[derive(Clone, PartialEq, Debug)]
pub struct Field {
    /// Entry name.
    pub name: Ident,
    /// Entry value.
    pub value: Expr,
    /// Source range covering the whole entry.
    pub span: Span,
}

impl Field {
    /// Builds a field entry.
    pub fn new(name: Ident, value: Expr, span: Span) -> Self {
        Self { name, value, span }
    }
}

/// A complete parsed `.ape` file.
#[derive(Clone, PartialEq, Debug)]
pub struct Program {
    /// Top-level items in source order.
    pub items: Vec<Item>,
    /// Source range covering the whole file.
    pub span: Span,
}

impl Program {
    /// Builds a program.
    pub fn new(items: Vec<Item>, span: Span) -> Self {
        Self { items, span }
    }

    /// Source range of the program.
    pub fn span(&self) -> Span {
        self.span
    }
}

/// Distinguishes differentiable values from trainable parameters.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TensorKind {
    /// The `tensor` keyword.
    Tensor,
    /// The `parameter` keyword.
    Parameter,
}

impl fmt::Display for TensorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorKind::Tensor => f.write_str("tensor"),
            TensorKind::Parameter => f.write_str("parameter"),
        }
    }
}

/// `tensor` / `parameter` declaration, for example
/// `parameter b : f32[2] = [0.0, 0.0]`.
#[derive(Clone, PartialEq, Debug)]
pub struct TensorDecl {
    /// Whether the binding is trainable.
    pub kind: TensorKind,
    /// Bound name.
    pub name: Ident,
    /// Optional explicit type; inferred when omitted.
    pub ty: Option<TypeExpr>,
    /// Initial value.
    pub value: Expr,
    /// Source range.
    pub span: Span,
}

/// `let` binding with optional type ascription.
#[derive(Clone, PartialEq, Debug)]
pub struct LetDecl {
    /// Bound name.
    pub name: Ident,
    /// Optional explicit type.
    pub ty: Option<TypeExpr>,
    /// Bound value.
    pub value: Expr,
    /// Source range.
    pub span: Span,
}

/// A function parameter.
#[derive(Clone, PartialEq, Debug)]
pub struct Param {
    /// Parameter name.
    pub name: Ident,
    /// Optional type annotation.
    pub ty: Option<TypeExpr>,
    /// Source range.
    pub span: Span,
}

/// `fn` definition.
#[derive(Clone, PartialEq, Debug)]
pub struct FnDecl {
    /// Function name.
    pub name: Ident,
    /// Ordered parameters.
    pub params: Vec<Param>,
    /// Optional return type.
    pub return_type: Option<TypeExpr>,
    /// Function body (usually a block expression).
    pub body: Expr,
    /// Source range.
    pub span: Span,
}

/// Memory tier selected by `memory <tier> <name> { ... }`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MemoryTier {
    /// `episodic`
    Episodic,
    /// `semantic`
    Semantic,
    /// `procedural`
    Procedural,
}

impl fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryTier::Episodic => f.write_str("episodic"),
            MemoryTier::Semantic => f.write_str("semantic"),
            MemoryTier::Procedural => f.write_str("procedural"),
        }
    }
}

/// `memory episodic ShortTermMem { dimension: 128, ... }`.
#[derive(Clone, PartialEq, Debug)]
pub struct MemoryDecl {
    /// Tier selector.
    pub tier: MemoryTier,
    /// Store name.
    pub name: Ident,
    /// Configuration entries.
    pub fields: Vec<Field>,
    /// Source range.
    pub span: Span,
}

/// `belief` declaration or Bayesian revision.
#[derive(Clone, PartialEq, Debug)]
pub enum BeliefStmt {
    /// `belief TargetThreat { proposition: ..., confidence: ..., ... }`.
    Declare {
        /// Belief name.
        name: Ident,
        /// Body entries.
        fields: Vec<Field>,
        /// Source range.
        span: Span,
    },
    /// `belief TargetThreat update { likelihood: ..., reliability: ... }`.
    Update {
        /// Belief being revised.
        name: Ident,
        /// Body entries.
        fields: Vec<Field>,
        /// Source range.
        span: Span,
    },
}

impl BeliefStmt {
    /// Name of the belief being declared or updated.
    pub fn name(&self) -> &Ident {
        match self {
            BeliefStmt::Declare { name, .. } | BeliefStmt::Update { name, .. } => name,
        }
    }

    /// Body entries.
    pub fn fields(&self) -> &[Field] {
        match self {
            BeliefStmt::Declare { fields, .. } | BeliefStmt::Update { fields, .. } => fields,
        }
    }

    /// Source range.
    pub fn span(&self) -> Span {
        match self {
            BeliefStmt::Declare { span, .. } | BeliefStmt::Update { span, .. } => *span,
        }
    }

    /// Whether this statement revises an existing belief.
    pub fn is_update(&self) -> bool {
        matches!(self, BeliefStmt::Update { .. })
    }
}

/// One mutation inside an `architecture` block.
#[derive(Clone, PartialEq, Debug)]
pub enum ArchOp {
    /// `add component "reflection_module"`.
    AddComponent {
        /// Component descriptor (usually a string literal).
        component: Expr,
        /// Source range.
        span: Span,
    },
    /// `expand memory "ShortTermMem" by 2000`.
    ExpandMemory {
        /// Memory store being grown.
        memory: Expr,
        /// Growth amount.
        by: Expr,
        /// Source range.
        span: Span,
    },
    /// `change routing "primary_llm -> fallback_llm"`.
    ChangeRouting {
        /// New routing rule.
        routing: Expr,
        /// Source range.
        span: Span,
    },
}

impl ArchOp {
    /// Source range of the operation.
    pub fn span(&self) -> Span {
        match self {
            ArchOp::AddComponent { span, .. }
            | ArchOp::ExpandMemory { span, .. }
            | ArchOp::ChangeRouting { span, .. } => *span,
        }
    }
}

/// `architecture DynamicCore { ... }` runtime self-modification block.
#[derive(Clone, PartialEq, Debug)]
pub struct ArchitectureBlock {
    /// Target architecture name.
    pub name: Ident,
    /// Ordered mutations.
    pub operations: Vec<ArchOp>,
    /// Source range.
    pub span: Span,
}

/// One member of a `society` block.
#[derive(Clone, PartialEq, Debug)]
pub enum SocietyMember {
    /// `agent Planner { role: "architect" }`.
    Agent {
        /// Agent name.
        name: Ident,
        /// Configuration entries.
        fields: Vec<Field>,
        /// Source range.
        span: Span,
    },
    /// `cluster MainCluster = [Planner, Executor]`.
    Cluster {
        /// Cluster name.
        name: Ident,
        /// Cluster members.
        members: Vec<Ident>,
        /// Source range.
        span: Span,
    },
}

impl SocietyMember {
    /// Member name.
    pub fn name(&self) -> &Ident {
        match self {
            SocietyMember::Agent { name, .. } | SocietyMember::Cluster { name, .. } => name,
        }
    }

    /// Source range.
    pub fn span(&self) -> Span {
        match self {
            SocietyMember::Agent { span, .. } | SocietyMember::Cluster { span, .. } => *span,
        }
    }
}

/// `society MultiAgentSwarm { ... }` orchestration block.
#[derive(Clone, PartialEq, Debug)]
pub struct SocietyBlock {
    /// Society name.
    pub name: Ident,
    /// Declared agents and clusters.
    pub members: Vec<SocietyMember>,
    /// Source range.
    pub span: Span,
}

/// A top-level (or block-level) statement.
#[derive(Clone, PartialEq, Debug)]
pub enum Item {
    /// `fn name(params) { ... }`
    Fn(FnDecl),
    /// `let name = value`
    Let(LetDecl),
    /// `tensor` / `parameter` declaration.
    Tensor(TensorDecl),
    /// `return expr?`
    Return {
        /// Optional returned value.
        value: Option<Expr>,
        /// Source range.
        span: Span,
    },
    /// `while condition body`
    While {
        /// Loop condition.
        condition: Expr,
        /// Loop body (usually a block).
        body: Expr,
        /// Source range.
        span: Span,
    },
    /// `for variable in iterable body`
    For {
        /// Loop variable.
        variable: Ident,
        /// Iterated expression.
        iterable: Expr,
        /// Loop body.
        body: Expr,
        /// Source range.
        span: Span,
    },
    /// `break`
    Break {
        /// Source range.
        span: Span,
    },
    /// `continue`
    Continue {
        /// Source range.
        span: Span,
    },
    /// A bare expression used as a statement.
    Expr(Expr),

    /// `optimizer adam { lr: 0.001 }`
    Optimizer {
        /// Optimizer family, for example `adam`.
        kind: Ident,
        /// Hyperparameters.
        settings: Vec<Field>,
        /// Source range.
        span: Span,
    },
    /// `train model on dataset`
    Train {
        /// Model or parameter being trained.
        model: Ident,
        /// Training dataset reference.
        dataset: Ident,
        /// Source range.
        span: Span,
    },
    /// `gradient loss with_respect_to W`
    Gradient {
        /// Scalar objective.
        target: Ident,
        /// Differentiated variable.
        with_respect_to: Ident,
        /// Source range.
        span: Span,
    },
    /// `distribution UserIntent { query: 0.65, ... }`
    Distribution {
        /// Distribution name.
        name: Ident,
        /// Category probabilities.
        categories: Vec<Field>,
        /// Source range.
        span: Span,
    },
    /// `belief` declaration or update.
    Belief(BeliefStmt),
    /// `cause A -> B`
    Cause {
        /// Antecedent node.
        from: Ident,
        /// Consequent node.
        to: Ident,
        /// Source range.
        span: Span,
    },
    /// `observe A = 85.5`
    Observe {
        /// Observed node.
        target: Ident,
        /// Observed value.
        value: Expr,
        /// Source range.
        span: Span,
    },
    /// `intervene A = value` (do-calculus).
    Intervene {
        /// Intervened node.
        target: Ident,
        /// Forced value.
        value: Expr,
        /// Source range.
        span: Span,
    },
    /// `counterfactual A if condition`
    Counterfactual {
        /// Node of interest.
        target: Ident,
        /// Hypothetical condition.
        condition: Expr,
        /// Source range.
        span: Span,
    },
    /// `simulate horizon = 50`
    Simulate {
        /// Simulated parameter.
        parameter: Ident,
        /// Simulation value.
        value: Expr,
        /// Source range.
        span: Span,
    },
    /// `memory <tier> name { ... }`
    Memory(MemoryDecl),
    /// `remember Store key = value ...`
    Remember {
        /// Target store.
        memory: Ident,
        /// Written entries.
        entries: Vec<Field>,
        /// Source range.
        span: Span,
    },
    /// `consolidate From -> To threshold = 0.8`
    Consolidate {
        /// Source tier.
        from: Ident,
        /// Destination tier.
        to: Ident,
        /// Optional similarity threshold.
        threshold: Option<Expr>,
        /// Source range.
        span: Span,
    },
    /// `recall Store query = expr top_k = 3`
    Recall {
        /// Queried store.
        memory: Ident,
        /// Query arguments.
        arguments: Vec<Field>,
        /// Source range.
        span: Span,
    },
    /// `plan RouteOptimization { goal: "...", horizon: 10 }`
    Plan {
        /// Plan name.
        name: Ident,
        /// Plan configuration.
        body: Vec<Field>,
        /// Source range.
        span: Span,
    },
    /// `self_model Name { ... }`
    SelfModel {
        /// Model name.
        name: Ident,
        /// Self-description entries.
        body: Vec<Field>,
        /// Source range.
        span: Span,
    },
    /// `architecture Name { ... }`
    Architecture(ArchitectureBlock),
    /// `society Name { ... }`
    Society(SocietyBlock),
    /// `route Request to Cluster budget { ... }`
    Route {
        /// Dispatched request.
        request: Ident,
        /// Target cluster.
        cluster: Ident,
        /// Budget constraints.
        budget: Vec<Field>,
        /// Source range.
        span: Span,
    },
}

impl Item {
    /// Source range of this item.
    pub fn span(&self) -> Span {
        match self {
            Item::Fn(decl) => decl.span,
            Item::Let(decl) => decl.span,
            Item::Tensor(decl) => decl.span,
            Item::Return { span, .. }
            | Item::While { span, .. }
            | Item::For { span, .. }
            | Item::Break { span }
            | Item::Continue { span }
            | Item::Optimizer { span, .. }
            | Item::Train { span, .. }
            | Item::Gradient { span, .. }
            | Item::Distribution { span, .. }
            | Item::Cause { span, .. }
            | Item::Observe { span, .. }
            | Item::Intervene { span, .. }
            | Item::Counterfactual { span, .. }
            | Item::Simulate { span, .. }
            | Item::Remember { span, .. }
            | Item::Consolidate { span, .. }
            | Item::Recall { span, .. }
            | Item::Plan { span, .. }
            | Item::SelfModel { span, .. }
            | Item::Route { span, .. } => *span,
            Item::Expr(expr) => expr.span(),
            Item::Belief(stmt) => stmt.span(),
            Item::Memory(decl) => decl.span,
            Item::Architecture(block) => block.span,
            Item::Society(block) => block.span,
        }
    }
}

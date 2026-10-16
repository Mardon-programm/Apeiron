//! IR instructions, including the cognitive primitive groups.

use apeiron_ast::{BinaryOp, MemoryTier, UnaryOp};

use crate::types::ElemType;
use crate::value::ValueId;

/// A compile-time constant.
#[derive(Clone, Debug, PartialEq)]
pub enum ConstValue {
    /// Integer constant.
    Int(i64),
    /// Floating point constant.
    Float(f64),
    /// Boolean constant.
    Bool(bool),
    /// String constant.
    Str(String),
    /// The unit value.
    Unit,
}

impl std::fmt::Display for ConstValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstValue::Int(value) => write!(f, "{value}"),
            ConstValue::Float(value) => write!(f, "{value}"),
            ConstValue::Bool(value) => write!(f, "{value}"),
            ConstValue::Str(value) => write!(f, "{value:?}"),
            ConstValue::Unit => f.write_str("()"),
        }
    }
}

/// A fully static tensor literal.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorLit {
    /// Element type.
    pub elem: ElemType,
    /// Static shape.
    pub shape: Vec<u64>,
    /// Flat row-major data.
    pub data: Vec<f64>,
}

impl TensorLit {
    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the literal has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Belief declaration or Bayesian update.
#[derive(Clone, Debug)]
pub enum BeliefInstr {
    /// `belief B { proposition, confidence }`.
    Declare {
        /// Belief name.
        name: String,
        /// The proposition being held.
        proposition: ValueId,
        /// Confidence in that proposition.
        confidence: ValueId,
    },
    /// `belief B update { likelihood, reliability }`.
    Update {
        /// Revised belief.
        name: String,
        /// Likelihood of the evidence.
        likelihood: ValueId,
        /// Reliability of the source.
        reliability: ValueId,
    },
}

/// Causal graph manipulation.
#[derive(Clone, Debug)]
pub enum CausalInstr {
    /// `cause A -> B`.
    Cause {
        /// Antecedent node.
        from: String,
        /// Consequent node.
        to: String,
    },
    /// `observe A = value`.
    Observe {
        /// Observed node.
        node: String,
        /// Observed value.
        value: ValueId,
    },
    /// `intervene A = value`.
    Intervene {
        /// Intervened node.
        node: String,
        /// Forced value.
        value: ValueId,
    },
    /// `counterfactual A if condition`.
    Counterfactual {
        /// Node of interest.
        node: String,
        /// Hypothetical condition.
        condition: ValueId,
    },
    /// `simulate parameter = value`.
    Simulate {
        /// Simulated parameter.
        parameter: String,
        /// Value.
        value: ValueId,
    },
}

/// Multi-tier memory operations.
#[derive(Clone, Debug)]
pub enum MemoryInstr {
    /// `memory <tier> Name { ... }`.
    Declare {
        /// Tier selector.
        tier: MemoryTier,
        /// Store name.
        name: String,
        /// Embedding dimension.
        dimension: ValueId,
        /// Optional capacity bound.
        capacity: Option<ValueId>,
        /// Optional retrieval metric name.
        metric: Option<ValueId>,
    },
    /// `remember Store k = v ...`.
    Remember {
        /// Target store.
        store: String,
        /// Written entries.
        entries: Vec<(String, ValueId)>,
    },
    /// `consolidate From -> To [threshold]`.
    Consolidate {
        /// Source store.
        from: String,
        /// Destination store.
        to: String,
        /// Optional similarity threshold.
        threshold: Option<ValueId>,
    },
    /// `recall Store key = value ...`.
    Recall {
        /// Queried store.
        store: String,
        /// Query arguments.
        arguments: Vec<(String, ValueId)>,
    },
}

/// Runtime architecture mutations.
#[derive(Clone, Debug)]
pub enum ArchInstr {
    /// `add component "..."`.
    AddComponent {
        /// Component descriptor.
        component: ValueId,
    },
    /// `expand memory "..." by n`.
    ExpandMemory {
        /// Store being grown.
        memory: String,
        /// Growth amount.
        by: ValueId,
    },
    /// `change routing "..."`.
    ChangeRouting {
        /// New routing rule.
        routing: ValueId,
    },
}

/// Society construction.
#[derive(Clone, Debug)]
pub enum SocietyInstr {
    /// `society Name { ... }`.
    DeclareSociety {
        /// Society name.
        name: String,
    },
    /// `agent Name { role }`.
    DeclareAgent {
        /// Owning society.
        society: String,
        /// Agent name.
        name: String,
        /// Agent role.
        role: ValueId,
    },
    /// `cluster Name = [members]`.
    DeclareCluster {
        /// Owning society.
        society: String,
        /// Cluster name.
        name: String,
        /// Member agent names.
        members: Vec<String>,
    },
}

/// The operation performed by an [`Instruction`].
#[derive(Clone, Debug)]
pub enum InstrKind {
    /// Materialise a constant.
    Const(ConstValue),
    /// Materialise a tensor literal.
    TensorLit(TensorLit),
    /// Reference a module global by name.
    GlobalRef {
        /// Global name.
        name: String,
    },
    /// Build a tensor from runtime element values.
    MakeTensor {
        /// Elements, row-major and possibly nested.
        elements: Vec<ValueId>,
    },
    /// Prefix operator.
    Unary {
        /// Operator.
        op: UnaryOp,
        /// Operand.
        operand: ValueId,
    },
    /// Infix arithmetic or comparison.
    Binary {
        /// Operator.
        op: BinaryOp,
        /// Left operand.
        lhs: ValueId,
        /// Right operand.
        rhs: ValueId,
    },
    /// Matrix multiplication.
    MatMul {
        /// Left operand.
        lhs: ValueId,
        /// Right operand.
        rhs: ValueId,
    },
    /// Call a function.
    Call {
        /// Callee name.
        callee: String,
        /// Positional arguments.
        args: Vec<ValueId>,
    },
    /// Dynamic indexing.
    Index {
        /// Indexed tensor.
        target: ValueId,
        /// Index value.
        index: ValueId,
    },
    /// Static extraction of the `index`-th element along axis 0.
    Extract {
        /// Source tensor.
        tensor: ValueId,
        /// Static lane.
        index: u64,
    },
    /// Static extent of one axis.
    Dimension {
        /// Source tensor.
        tensor: ValueId,
        /// Axis (0-based).
        axis: u32,
    },
    /// Reverse-mode derivative of `target` with respect to `wrt`.
    Gradient {
        /// Scalar objective.
        target: ValueId,
        /// Differentiated value.
        wrt: ValueId,
    },
    /// Belief operation.
    Belief(BeliefInstr),
    /// Categorical distribution declaration.
    Distribution {
        /// Distribution name.
        name: String,
        /// Category probabilities.
        categories: Vec<(String, ValueId)>,
    },
    /// Causal operation.
    Causal(CausalInstr),
    /// Memory operation.
    Memory(MemoryInstr),
    /// Architecture mutation.
    Architecture(ArchInstr),
    /// Society construction.
    Society(SocietyInstr),
    /// Plan declaration.
    Plan {
        /// Plan name.
        name: String,
        /// Plan goal.
        goal: ValueId,
    },
    /// Optimizer construction.
    Optimizer {
        /// Optimizer family.
        kind: String,
        /// Hyperparameters.
        settings: Vec<(String, ValueId)>,
    },
    /// Training step.
    Train {
        /// Model or parameter.
        model: ValueId,
        /// Dataset reference.
        dataset: ValueId,
    },
    /// Request routing.
    Route {
        /// Dispatched request.
        request: ValueId,
        /// Target cluster.
        cluster: String,
        /// Budget constraints.
        budget: Vec<(String, ValueId)>,
    },
}

impl InstrKind {
    /// Whether the instruction has observable effects that dead-code
    /// elimination must preserve.
    pub fn is_side_effecting(&self) -> bool {
        matches!(
            self,
            InstrKind::Belief(_)
                | InstrKind::Causal(_)
                | InstrKind::Memory(_)
                | InstrKind::Architecture(_)
                | InstrKind::Society(_)
                | InstrKind::Optimizer { .. }
                | InstrKind::Train { .. }
                | InstrKind::Route { .. }
                | InstrKind::Plan { .. }
        )
    }

    /// Whether the instruction always produces a result value.
    pub fn produces_value(&self) -> bool {
        !self.is_side_effecting()
    }

    /// Every value read by this instruction.
    pub fn operands(&self) -> Vec<ValueId> {
        let mut out = Vec::new();
        match self {
            InstrKind::Const(_) | InstrKind::TensorLit(_) | InstrKind::GlobalRef { .. } => {}
            InstrKind::MakeTensor { elements } => out.extend(elements.iter().copied()),
            InstrKind::Unary { operand, .. } => out.push(*operand),
            InstrKind::Binary { lhs, rhs, .. } | InstrKind::MatMul { lhs, rhs } => {
                out.push(*lhs);
                out.push(*rhs);
            }
            InstrKind::Call { args, .. } => out.extend(args.iter().copied()),
            InstrKind::Index { target, index } => {
                out.push(*target);
                out.push(*index);
            }
            InstrKind::Extract { tensor, .. } | InstrKind::Dimension { tensor, .. } => {
                out.push(*tensor)
            }
            InstrKind::Gradient { target, wrt } => {
                out.push(*target);
                out.push(*wrt);
            }
            InstrKind::Belief(inner) => match inner {
                BeliefInstr::Declare {
                    proposition,
                    confidence,
                    ..
                } => {
                    out.push(*proposition);
                    out.push(*confidence);
                }
                BeliefInstr::Update {
                    likelihood,
                    reliability,
                    ..
                } => {
                    out.push(*likelihood);
                    out.push(*reliability);
                }
            },
            InstrKind::Distribution { categories, .. } => {
                out.extend(categories.iter().map(|(_, value)| *value));
            }
            InstrKind::Causal(inner) => match inner {
                CausalInstr::Cause { .. } => {}
                CausalInstr::Observe { value, .. }
                | CausalInstr::Intervene { value, .. }
                | CausalInstr::Simulate { value, .. } => out.push(*value),
                CausalInstr::Counterfactual { condition, .. } => out.push(*condition),
            },
            InstrKind::Memory(inner) => match inner {
                MemoryInstr::Declare {
                    dimension,
                    capacity,
                    metric,
                    ..
                } => {
                    out.push(*dimension);
                    out.extend(capacity.iter().copied());
                    out.extend(metric.iter().copied());
                }
                MemoryInstr::Remember { entries, .. } => {
                    out.extend(entries.iter().map(|(_, value)| *value))
                }
                MemoryInstr::Consolidate { threshold, .. } => out.extend(threshold.iter().copied()),
                MemoryInstr::Recall { arguments, .. } => {
                    out.extend(arguments.iter().map(|(_, value)| *value))
                }
            },
            InstrKind::Architecture(inner) => match inner {
                ArchInstr::AddComponent { component }
                | ArchInstr::ChangeRouting { routing: component } => out.push(*component),
                ArchInstr::ExpandMemory { by, .. } => out.push(*by),
            },
            InstrKind::Society(inner) => match inner {
                SocietyInstr::DeclareSociety { .. } | SocietyInstr::DeclareCluster { .. } => {}
                SocietyInstr::DeclareAgent { role, .. } => out.push(*role),
            },
            InstrKind::Plan { goal, .. } => out.push(*goal),
            InstrKind::Optimizer { settings, .. } => {
                out.extend(settings.iter().map(|(_, value)| *value))
            }
            InstrKind::Train { model, dataset } => {
                out.push(*model);
                out.push(*dataset);
            }
            InstrKind::Route {
                request, budget, ..
            } => {
                out.push(*request);
                out.extend(budget.iter().map(|(_, value)| *value));
            }
        }
        out
    }
}

/// One instruction with its optional result and source anchor.
#[derive(Clone, Debug)]
pub struct Instruction {
    /// Result value, absent for effect-only operations.
    pub result: Option<ValueId>,
    /// The operation.
    pub kind: InstrKind,
    /// Originating source range, when available.
    pub span: Option<apeiron_core::span::Span>,
}

impl Instruction {
    /// Builds an instruction that yields a value.
    pub fn value(result: ValueId, kind: InstrKind) -> Self {
        Self {
            result: Some(result),
            kind,
            span: None,
        }
    }

    /// Builds an effect-only instruction.
    pub fn effect(kind: InstrKind) -> Self {
        Self {
            result: None,
            kind,
            span: None,
        }
    }

    /// Attaches a source range.
    pub fn with_span(mut self, span: apeiron_core::span::Span) -> Self {
        self.span = Some(span);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_lit_length() {
        let literal = TensorLit {
            elem: ElemType::F32,
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        assert_eq!(literal.len(), 4);
        assert!(!literal.is_empty());
    }

    #[test]
    fn side_effecting_classification() {
        assert!(InstrKind::Causal(CausalInstr::Cause {
            from: "A".into(),
            to: "B".into()
        })
        .is_side_effecting());
        assert!(!InstrKind::Const(ConstValue::Int(1)).is_side_effecting());
        assert!(InstrKind::Const(ConstValue::Int(1)).produces_value());
    }
}

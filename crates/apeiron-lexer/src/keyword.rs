//! Reserved words of the Apeiron language.
//!
//! The set is intentionally domain heavy: tensors, epistemic beliefs,
//! causal reasoning, memory tiers, runtime architecture mutation, and
//! multi-agent societies all contribute first-class keywords. Anything
//! that is not in this table is lexed as an ordinary identifier.

use std::fmt;

/// Every reserved word in Apeiron.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Keyword {
    // ── control flow & declarations ─────────────────────────────────
    /// `let` — binding with optional type ascription.
    Let,
    /// `fn` — function definition.
    Fn,
    /// `return` — early exit.
    Return,
    /// `if` — conditional expression.
    If,
    /// `else` — alternative branch.
    Else,
    /// `while` — pre-test loop.
    While,
    /// `for` — iterator loop.
    For,
    /// `in` — `for x in xs` binder.
    In,
    /// `break` — loop exit.
    Break,
    /// `continue` — next loop iteration.
    Continue,
    /// `true` — boolean literal.
    True,
    /// `false` — boolean literal.
    False,

    // ── tensors & training ──────────────────────────────────────────
    /// `tensor` — differentiable value declaration.
    Tensor,
    /// `parameter` — trainable tensor declaration.
    Parameter,
    /// `gradient` — reverse-mode differentiation request.
    Gradient,
    /// `with_respect_to` — differentiation target selector.
    WithRespectTo,
    /// `optimizer` — optimizer configuration block.
    Optimizer,
    /// `train` — training invocation.
    Train,
    /// `on` — dataset selector for `train`.
    On,

    // ── epistemic beliefs ───────────────────────────────────────────
    /// `distribution` — categorical probability declaration.
    Distribution,
    /// `belief` — epistemic belief declaration or update.
    Belief,
    /// `proposition` — belief body field.
    Proposition,
    /// `confidence` — belief body field.
    Confidence,
    /// `because` — causal justification field.
    Because,
    /// `update` — Bayesian belief revision marker.
    Update,
    /// `likelihood` — update body field.
    Likelihood,
    /// `reliability` — update body field.
    Reliability,

    // ── causal engine ───────────────────────────────────────────────
    /// `cause` — directed edge in the causal graph.
    Cause,
    /// `observe` — condition a node on a value.
    Observe,
    /// `intervene` — do-calculus intervention.
    Intervene,
    /// `counterfactual` — what-if query.
    Counterfactual,
    /// `simulate` — forward simulation driver.
    Simulate,
    /// `horizon` — simulation length parameter.
    Horizon,

    // ── memory & cognition ──────────────────────────────────────────
    /// `memory` — multi-tier memory declaration.
    Memory,
    /// `episodic` — episodic memory tier.
    Episodic,
    /// `semantic` — semantic memory tier.
    Semantic,
    /// `procedural` — procedural memory tier.
    Procedural,
    /// `remember` — write into a memory store.
    Remember,
    /// `consolidate` — migrate memories between tiers.
    Consolidate,
    /// `threshold` — consolidation similarity cutoff.
    Threshold,
    /// `recall` — similarity search over a memory store.
    Recall,
    /// `plan` — cognitive planning block.
    Plan,
    /// `self_model` — agent self-description block.
    SelfModel,

    // ── architectural self-modification ─────────────────────────────
    /// `architecture` — runtime graph mutation block.
    Architecture,
    /// `add` — insert a component.
    Add,
    /// `component` — component selector.
    Component,
    /// `expand` — grow a subsystem.
    Expand,
    /// `by` — expansion amount selector.
    By,
    /// `change` — retarget wiring.
    Change,
    /// `routing` — routing table selector.
    Routing,

    // ── multi-agent societies ───────────────────────────────────────
    /// `society` — agent orchestration block.
    Society,
    /// `agent` — single agent declaration.
    Agent,
    /// `cluster` — static agent group.
    Cluster,
    /// `route` — budget-aware request dispatch.
    Route,
    /// `to` — routing destination selector.
    To,
    /// `budget` — economic budget block.
    Budget,
}

impl Keyword {
    /// Canonical identifier spelling of every keyword, in enum order.
    pub const ALL: [Keyword; 56] = [
        Keyword::Let,
        Keyword::Fn,
        Keyword::Return,
        Keyword::If,
        Keyword::Else,
        Keyword::While,
        Keyword::For,
        Keyword::In,
        Keyword::Break,
        Keyword::Continue,
        Keyword::True,
        Keyword::False,
        Keyword::Tensor,
        Keyword::Parameter,
        Keyword::Gradient,
        Keyword::WithRespectTo,
        Keyword::Optimizer,
        Keyword::Train,
        Keyword::On,
        Keyword::Distribution,
        Keyword::Belief,
        Keyword::Proposition,
        Keyword::Confidence,
        Keyword::Because,
        Keyword::Update,
        Keyword::Likelihood,
        Keyword::Reliability,
        Keyword::Cause,
        Keyword::Observe,
        Keyword::Intervene,
        Keyword::Counterfactual,
        Keyword::Simulate,
        Keyword::Horizon,
        Keyword::Memory,
        Keyword::Episodic,
        Keyword::Semantic,
        Keyword::Procedural,
        Keyword::Remember,
        Keyword::Consolidate,
        Keyword::Threshold,
        Keyword::Recall,
        Keyword::Plan,
        Keyword::SelfModel,
        Keyword::Architecture,
        Keyword::Add,
        Keyword::Component,
        Keyword::Expand,
        Keyword::By,
        Keyword::Change,
        Keyword::Routing,
        Keyword::Society,
        Keyword::Agent,
        Keyword::Cluster,
        Keyword::Route,
        Keyword::To,
        Keyword::Budget,
    ];

    /// Resolves an identifier to a keyword, if it is reserved.
    pub fn from_ident(text: &str) -> Option<Keyword> {
        let kw = match text {
            "let" => Keyword::Let,
            "fn" => Keyword::Fn,
            "return" => Keyword::Return,
            "if" => Keyword::If,
            "else" => Keyword::Else,
            "while" => Keyword::While,
            "for" => Keyword::For,
            "in" => Keyword::In,
            "break" => Keyword::Break,
            "continue" => Keyword::Continue,
            "true" => Keyword::True,
            "false" => Keyword::False,
            "tensor" => Keyword::Tensor,
            "parameter" => Keyword::Parameter,
            "gradient" => Keyword::Gradient,
            "with_respect_to" => Keyword::WithRespectTo,
            "optimizer" => Keyword::Optimizer,
            "train" => Keyword::Train,
            "on" => Keyword::On,
            "distribution" => Keyword::Distribution,
            "belief" => Keyword::Belief,
            "proposition" => Keyword::Proposition,
            "confidence" => Keyword::Confidence,
            "because" => Keyword::Because,
            "update" => Keyword::Update,
            "likelihood" => Keyword::Likelihood,
            "reliability" => Keyword::Reliability,
            "cause" => Keyword::Cause,
            "observe" => Keyword::Observe,
            "intervene" => Keyword::Intervene,
            "counterfactual" => Keyword::Counterfactual,
            "simulate" => Keyword::Simulate,
            "horizon" => Keyword::Horizon,
            "memory" => Keyword::Memory,
            "episodic" => Keyword::Episodic,
            "semantic" => Keyword::Semantic,
            "procedural" => Keyword::Procedural,
            "remember" => Keyword::Remember,
            "consolidate" => Keyword::Consolidate,
            "threshold" => Keyword::Threshold,
            "recall" => Keyword::Recall,
            "plan" => Keyword::Plan,
            "self_model" => Keyword::SelfModel,
            "architecture" => Keyword::Architecture,
            "add" => Keyword::Add,
            "component" => Keyword::Component,
            "expand" => Keyword::Expand,
            "by" => Keyword::By,
            "change" => Keyword::Change,
            "routing" => Keyword::Routing,
            "society" => Keyword::Society,
            "agent" => Keyword::Agent,
            "cluster" => Keyword::Cluster,
            "route" => Keyword::Route,
            "to" => Keyword::To,
            "budget" => Keyword::Budget,
            _ => return None,
        };
        Some(kw)
    }

    /// Canonical source spelling of this keyword.
    pub const fn as_str(self) -> &'static str {
        match self {
            Keyword::Let => "let",
            Keyword::Fn => "fn",
            Keyword::Return => "return",
            Keyword::If => "if",
            Keyword::Else => "else",
            Keyword::While => "while",
            Keyword::For => "for",
            Keyword::In => "in",
            Keyword::Break => "break",
            Keyword::Continue => "continue",
            Keyword::True => "true",
            Keyword::False => "false",
            Keyword::Tensor => "tensor",
            Keyword::Parameter => "parameter",
            Keyword::Gradient => "gradient",
            Keyword::WithRespectTo => "with_respect_to",
            Keyword::Optimizer => "optimizer",
            Keyword::Train => "train",
            Keyword::On => "on",
            Keyword::Distribution => "distribution",
            Keyword::Belief => "belief",
            Keyword::Proposition => "proposition",
            Keyword::Confidence => "confidence",
            Keyword::Because => "because",
            Keyword::Update => "update",
            Keyword::Likelihood => "likelihood",
            Keyword::Reliability => "reliability",
            Keyword::Cause => "cause",
            Keyword::Observe => "observe",
            Keyword::Intervene => "intervene",
            Keyword::Counterfactual => "counterfactual",
            Keyword::Simulate => "simulate",
            Keyword::Horizon => "horizon",
            Keyword::Memory => "memory",
            Keyword::Episodic => "episodic",
            Keyword::Semantic => "semantic",
            Keyword::Procedural => "procedural",
            Keyword::Remember => "remember",
            Keyword::Consolidate => "consolidate",
            Keyword::Threshold => "threshold",
            Keyword::Recall => "recall",
            Keyword::Plan => "plan",
            Keyword::SelfModel => "self_model",
            Keyword::Architecture => "architecture",
            Keyword::Add => "add",
            Keyword::Component => "component",
            Keyword::Expand => "expand",
            Keyword::By => "by",
            Keyword::Change => "change",
            Keyword::Routing => "routing",
            Keyword::Society => "society",
            Keyword::Agent => "agent",
            Keyword::Cluster => "cluster",
            Keyword::Route => "route",
            Keyword::To => "to",
            Keyword::Budget => "budget",
        }
    }
}

impl fmt::Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_keywords_round_trip() {
        for kw in Keyword::ALL {
            let text = kw.as_str();
            assert_eq!(
                Keyword::from_ident(text),
                Some(kw),
                "round trip failed for {text}"
            );
            assert_eq!(text, Keyword::from_ident(text).unwrap().as_str());
        }
    }

    #[test]
    fn all_keywords_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for kw in Keyword::ALL {
            assert!(
                seen.insert(kw.as_str()),
                "duplicate keyword spelling: {}",
                kw.as_str()
            );
        }
        assert_eq!(seen.len(), Keyword::ALL.len());
    }

    #[test]
    fn non_keywords_are_rejected() {
        assert_eq!(Keyword::from_ident("tensorx"), None);
        assert_eq!(Keyword::from_ident("Tensor"), None);
        assert_eq!(Keyword::from_ident(""), None);
        assert_eq!(Keyword::from_ident("vector_query"), None);
    }

    #[test]
    fn display_matches_as_str() {
        assert_eq!(Keyword::WithRespectTo.to_string(), "with_respect_to");
        assert_eq!(Keyword::Counterfactual.to_string(), "counterfactual");
    }
}

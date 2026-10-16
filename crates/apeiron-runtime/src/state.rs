//! The cognitive state a program builds up as it runs.
//!
//! Everything a module declares — globals, memories, beliefs, distributions,
//! plans, architecture mutations, societies, optimizers, and training runs —
//! lives here so a caller can inspect it after execution.

use std::collections::{BTreeMap, HashMap};

use crate::causal::CausalModel;
use crate::memory::MemoryStore;
use crate::value::Value;

/// A probabilistic belief about a proposition.
#[derive(Clone, Debug, PartialEq)]
pub struct Belief {
    /// The proposition held.
    pub proposition: String,
    /// Current confidence in `[0, 1]`.
    pub confidence: f64,
    /// Recorded evidence descriptions.
    pub evidence: Vec<String>,
    /// How many updates have been applied.
    pub updates: usize,
}

impl Belief {
    /// Creates a belief with an initial confidence.
    pub fn new(proposition: impl Into<String>, confidence: f64) -> Self {
        Self {
            proposition: proposition.into(),
            confidence: confidence.clamp(0.0, 1.0),
            evidence: Vec::new(),
            updates: 0,
        }
    }

    /// Applies a Bayesian-flavoured update from a likelihood and source
    /// reliability, returning the posterior confidence.
    pub fn update(&mut self, likelihood: f64, reliability: f64) -> f64 {
        let likelihood = likelihood.clamp(0.0, 1.0);
        let reliability = reliability.clamp(0.0, 1.0);
        let prior = self.confidence;
        let denominator = prior * likelihood + (1.0 - prior) * (1.0 - likelihood);
        let posterior = if denominator == 0.0 {
            prior
        } else {
            prior * likelihood / denominator
        };
        self.confidence = (reliability * posterior + (1.0 - reliability) * prior).clamp(0.0, 1.0);
        self.updates += 1;
        self.evidence.push(format!(
            "likelihood={likelihood:.3}, reliability={reliability:.3}"
        ));
        self.confidence
    }
}

/// A categorical probability distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct Distribution {
    /// Category name to probability.
    pub categories: Vec<(String, f64)>,
}

impl Distribution {
    /// Builds a distribution, normalising the weights.
    pub fn new(categories: Vec<(String, f64)>) -> Self {
        let mut distribution = Self { categories };
        distribution.normalize();
        distribution
    }

    /// Rescales the weights so they sum to one.
    pub fn normalize(&mut self) {
        let total: f64 = self.categories.iter().map(|(_, weight)| weight).sum();
        if total != 0.0 {
            for (_, weight) in &mut self.categories {
                *weight /= total;
            }
        }
    }

    /// Probability of a category, if present.
    pub fn probability(&self, category: &str) -> Option<f64> {
        self.categories
            .iter()
            .find(|(name, _)| name == category)
            .map(|(_, weight)| *weight)
    }

    /// The most probable category, if any.
    pub fn argmax(&self) -> Option<&str> {
        self.categories
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, _)| name.as_str())
    }
}

/// A goal-directed plan.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// Plan name.
    pub name: String,
    /// Declared goal.
    pub goal: String,
}

/// Runtime architecture mutations.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ArchitectureState {
    /// Added component descriptors.
    pub components: Vec<String>,
    /// Memory expansions as `(store, amount)`.
    pub memory_expansions: Vec<(String, i64)>,
    /// Routing rules that were installed.
    pub routing: Vec<String>,
}

/// Agents and clusters grouped into societies.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SocietyState {
    /// Declared society names.
    pub societies: Vec<String>,
    /// Agent name to role.
    pub agents: BTreeMap<String, String>,
    /// Cluster name to member agent names.
    pub clusters: BTreeMap<String, Vec<String>>,
}

impl SocietyState {
    /// Adds an agent.
    pub fn add_agent(&mut self, name: impl Into<String>, role: impl Into<String>) {
        self.agents.insert(name.into(), role.into());
    }

    /// Adds a cluster.
    pub fn add_cluster(&mut self, name: impl Into<String>, members: Vec<String>) {
        self.clusters.insert(name.into(), members);
    }
}

/// A configured optimizer.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizerState {
    /// Optimizer family (adam, sgd, ...).
    pub kind: String,
    /// Hyperparameters.
    pub settings: BTreeMap<String, f64>,
}

/// A training invocation.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingRun {
    /// Model or parameter being trained.
    pub model: String,
    /// Dataset reference.
    pub dataset: String,
}

/// The complete runtime state.
#[derive(Clone, Debug, Default)]
pub struct RuntimeState {
    /// Module globals by name.
    pub globals: HashMap<String, Value>,
    /// Memory stores by name.
    pub memories: BTreeMap<String, MemoryStore>,
    /// Beliefs by name.
    pub beliefs: BTreeMap<String, Belief>,
    /// Distributions by name.
    pub distributions: BTreeMap<String, Distribution>,
    /// The causal model.
    pub causal: CausalModel,
    /// Plans by name.
    pub plans: BTreeMap<String, Plan>,
    /// Architecture mutations.
    pub architecture: ArchitectureState,
    /// Societies, agents, and clusters.
    pub society: SocietyState,
    /// Optimizers in creation order.
    pub optimizers: Vec<OptimizerState>,
    /// Training runs in creation order.
    pub training: Vec<TrainingRun>,
    /// Result of the most recent `recall`.
    pub last_recall: Option<Vec<(f64, Value)>>,
}

impl RuntimeState {
    /// Creates an empty state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Renders a compact, human-readable summary of everything the state
    /// holds. Sections with nothing to report are omitted; an entirely empty
    /// state renders as the empty string.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();

        if !self.globals.is_empty() {
            lines.push(format!("globals        : {}", self.globals.len()));
        }
        if !self.beliefs.is_empty() {
            let beliefs = self
                .beliefs
                .iter()
                .map(|(name, belief)| format!("{name} (p={:.3})", belief.confidence))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("beliefs        : {beliefs}"));
        }
        if !self.distributions.is_empty() {
            let distributions = self
                .distributions
                .iter()
                .map(|(name, distribution)| {
                    format!("{name} -> {}", distribution.argmax().unwrap_or("(empty)"))
                })
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("distributions  : {distributions}"));
        }
        if !self.memories.is_empty() {
            let memories = self
                .memories
                .iter()
                .map(|(name, store)| {
                    format!(
                        "{name} [{}, dim={}, entries={}]",
                        store.tier,
                        store.dimension,
                        store.len()
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            lines.push(format!("memories       : {memories}"));
        }
        if self.causal.edge_count() > 0
            || !self.causal.observations().is_empty()
            || !self.causal.interventions().is_empty()
        {
            lines.push(format!(
                "causal         : {} edges, {} observations, {} interventions, {} counterfactuals",
                self.causal.edge_count(),
                self.causal.observations().len(),
                self.causal.interventions().len(),
                self.causal.counterfactuals().len()
            ));
        }
        if !self.plans.is_empty() {
            let plans = self.plans.keys().cloned().collect::<Vec<_>>().join(", ");
            lines.push(format!("plans          : {plans}"));
        }
        if !self.architecture.components.is_empty()
            || !self.architecture.memory_expansions.is_empty()
            || !self.architecture.routing.is_empty()
        {
            lines.push(format!(
                "architecture   : {} components, {} expansions, {} routing rules",
                self.architecture.components.len(),
                self.architecture.memory_expansions.len(),
                self.architecture.routing.len()
            ));
        }
        if !self.society.societies.is_empty() {
            lines.push(format!(
                "society        : {} [{} agents, {} clusters]",
                self.society.societies.join(", "),
                self.society.agents.len(),
                self.society.clusters.len()
            ));
        }
        if !self.optimizers.is_empty() {
            let optimizers = self
                .optimizers
                .iter()
                .map(|optimizer| optimizer.kind.clone())
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("optimizers     : {optimizers}"));
        }
        if !self.training.is_empty() {
            lines.push(format!("training runs  : {}", self.training.len()));
        }
        if let Some(recall) = &self.last_recall {
            lines.push(format!("last recall    : {} hits", recall.len()));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn belief_update_raises_confidence_with_likely_evidence() {
        let mut belief = Belief::new("threat", 0.42);
        let posterior = belief.update(0.88, 0.95);
        assert!(posterior > 0.42);
        assert_eq!(belief.updates, 1);
    }

    #[test]
    fn distribution_normalises() {
        let distribution = Distribution::new(vec![("a".into(), 2.0), ("b".into(), 2.0)]);
        assert!((distribution.probability("a").unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn society_records_agents_and_clusters() {
        let mut society = SocietyState::default();
        society.add_agent("Planner", "architect");
        society.add_cluster("MainCluster", vec!["Planner".into()]);
        assert_eq!(society.agents["Planner"], "architect");
        assert_eq!(society.clusters["MainCluster"], vec!["Planner"]);
    }
}

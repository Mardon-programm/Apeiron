//! A simple structural causal model.
//!
//! Nodes form a directed graph of `cause A -> B` edges. Observations and
//! interventions attach numeric values to nodes, and counterfactuals record
//! hypothetical conditions together with the descendants they would disturb.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

/// A recorded counterfactual query.
#[derive(Clone, Debug, PartialEq)]
pub struct Counterfactual {
    /// Node of interest.
    pub node: String,
    /// Whether the hypothetical condition held.
    pub condition: bool,
}

/// Directed acyclic-ish causal graph with evidence.
#[derive(Clone, Debug, Default)]
pub struct CausalModel {
    edges: BTreeSet<(String, String)>,
    observations: HashMap<String, f64>,
    interventions: HashMap<String, f64>,
    simulations: Vec<(String, f64)>,
    counterfactuals: Vec<Counterfactual>,
}

impl CausalModel {
    /// Creates an empty model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a causal edge.
    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>) {
        self.edges.insert((from.into(), to.into()));
    }

    /// Records an observation.
    pub fn observe(&mut self, node: impl Into<String>, value: f64) {
        self.observations.insert(node.into(), value);
    }

    /// Records an intervention (a `do` operation).
    pub fn intervene(&mut self, node: impl Into<String>, value: f64) {
        self.interventions.insert(node.into(), value);
    }

    /// Records a simulation parameter.
    pub fn simulate(&mut self, parameter: impl Into<String>, value: f64) {
        self.simulations.push((parameter.into(), value));
    }

    /// Records a counterfactual.
    pub fn record_counterfactual(&mut self, node: impl Into<String>, condition: bool) {
        self.counterfactuals.push(Counterfactual {
            node: node.into(),
            condition,
        });
    }

    /// Every edge, lexicographically ordered.
    pub fn edges(&self) -> impl Iterator<Item = &(String, String)> {
        self.edges.iter()
    }

    /// Number of causal edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Observations recorded so far.
    pub fn observations(&self) -> &HashMap<String, f64> {
        &self.observations
    }

    /// Interventions recorded so far.
    pub fn interventions(&self) -> &HashMap<String, f64> {
        &self.interventions
    }

    /// Simulations recorded so far.
    pub fn simulations(&self) -> &[(String, f64)] {
        &self.simulations
    }

    /// Counterfactuals recorded so far.
    pub fn counterfactuals(&self) -> &[Counterfactual] {
        &self.counterfactuals
    }

    /// Direct parents of a node.
    pub fn parents(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|(_, to)| to == node)
            .map(|(from, _)| from.clone())
            .collect()
    }

    /// Direct children of a node.
    pub fn children(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|(from, _)| from == node)
            .map(|(_, to)| to.clone())
            .collect()
    }

    /// Whether `from` can reach `to` along causal edges.
    pub fn is_ancestor(&self, from: &str, to: &str) -> bool {
        let mut visited: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<&str> = VecDeque::new();
        queue.push_back(from);
        visited.insert(from);
        while let Some(node) = queue.pop_front() {
            for (edge_from, edge_to) in &self.edges {
                if edge_from == node {
                    if edge_to == to {
                        return true;
                    }
                    if visited.insert(edge_to) {
                        queue.push_back(edge_to);
                    }
                }
            }
        }
        false
    }

    /// All nodes reachable from `node` (its effect closure).
    pub fn descendants(&self, node: &str) -> BTreeSet<String> {
        let mut visited: BTreeSet<String> = BTreeSet::new();
        let mut queue: VecDeque<String> = self.children(node).into();
        while let Some(current) = queue.pop_front() {
            if visited.insert(current.clone()) {
                queue.extend(self.children(&current));
            }
        }
        visited
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_edges_and_ancestry() {
        let mut model = CausalModel::new();
        model.add_edge("Temperature", "Pressure");
        model.add_edge("Pressure", "ValveState");
        assert!(model.is_ancestor("Temperature", "ValveState"));
        assert!(!model.is_ancestor("ValveState", "Temperature"));
        assert_eq!(model.edge_count(), 2);
    }

    #[test]
    fn descendants_reach_closure() {
        let mut model = CausalModel::new();
        model.add_edge("A", "B");
        model.add_edge("B", "C");
        model.add_edge("A", "D");
        let descendants = model.descendants("A");
        assert!(descendants.contains("B"));
        assert!(descendants.contains("C"));
        assert!(descendants.contains("D"));
    }

    #[test]
    fn records_evidence() {
        let mut model = CausalModel::new();
        model.observe("Temperature", 85.5);
        model.intervene("ValveState", 1.0);
        model.simulate("horizon", 50.0);
        model.record_counterfactual("ValveState", true);
        assert_eq!(model.observations()["Temperature"], 85.5);
        assert_eq!(model.interventions()["ValveState"], 1.0);
        assert_eq!(model.simulations().len(), 1);
        assert_eq!(model.counterfactuals().len(), 1);
    }
}

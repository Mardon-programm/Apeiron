//! Multi-tier associative memory stores.
//!
//! A store holds embedding-keyed entries. Retrieval ranks entries by a
//! similarity metric and returns the top matches. Consolidation copies the
//! novel entries of one store into another, which is how the runtime models
//! short-term to long-term transfer.

use apeiron_ast::MemoryTier;

use crate::value::Value;

/// Similarity metric used for retrieval.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Metric {
    /// Cosine similarity.
    Cosine,
    /// Negative Euclidean distance (higher is closer).
    Euclidean,
    /// Raw dot product.
    Dot,
}

impl Metric {
    /// Parses a metric name, defaulting to cosine.
    pub fn parse(name: &str) -> Metric {
        match name.to_ascii_lowercase().as_str() {
            "euclidean" | "l2" => Metric::Euclidean,
            "dot" | "inner" => Metric::Dot,
            _ => Metric::Cosine,
        }
    }

    /// Similarity of two equally-sized vectors.
    pub fn similarity(self, a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        match self {
            Metric::Dot => (0..len).map(|i| a[i] * b[i]).sum(),
            Metric::Cosine => {
                let dot: f64 = (0..len).map(|i| a[i] * b[i]).sum();
                let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
                let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    0.0
                } else {
                    dot / (norm_a * norm_b)
                }
            }
            Metric::Euclidean => {
                let distance: f64 = (0..len).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt();
                -distance
            }
        }
    }
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Metric::Cosine => "cosine",
            Metric::Euclidean => "euclidean",
            Metric::Dot => "dot",
        })
    }
}

/// One stored association: an embedding key and its payload.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryEntry {
    /// Embedding key.
    pub key: Vec<f64>,
    /// Stored value.
    pub value: Value,
}

/// A keyed, capacity-bounded memory store.
#[derive(Clone, Debug)]
pub struct MemoryStore {
    /// Store name.
    pub name: String,
    /// Storage tier.
    pub tier: MemoryTier,
    /// Declared embedding dimension (`0` means unspecified).
    pub dimension: usize,
    /// Maximum number of entries (`0` means unbounded).
    pub capacity: usize,
    /// Retrieval metric.
    pub metric: Metric,
    /// Entries, oldest first.
    pub entries: Vec<MemoryEntry>,
}

impl MemoryStore {
    /// Creates an empty store.
    pub fn new(name: impl Into<String>, tier: MemoryTier, dimension: usize) -> Self {
        Self {
            name: name.into(),
            tier,
            dimension,
            capacity: 0,
            metric: Metric::Cosine,
            entries: Vec::new(),
        }
    }

    /// Sets the capacity.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Sets the retrieval metric.
    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Inserts an association, replacing an identical key and evicting the
    /// oldest entry when over capacity.
    pub fn remember(&mut self, key: Vec<f64>, value: Value) {
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|entry| entry.key.len() == key.len() && entry.key == key)
        {
            existing.value = value;
            return;
        }
        if self.capacity > 0 && self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }
        self.entries.push(MemoryEntry { key, value });
    }

    /// Ranks entries against a query and returns the best `top_k` values with
    /// their scores, highest score first.
    pub fn recall(&self, query: &[f64], top_k: usize) -> Vec<(f64, Value)> {
        let mut scored: Vec<(f64, Value)> = self
            .entries
            .iter()
            .map(|entry| {
                (
                    self.metric.similarity(query, &entry.key),
                    entry.value.clone(),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Copies entries into `destination` whose nearest neighbour there scores
    /// below `threshold`, returning how many were transferred.
    pub fn consolidate_into(&self, destination: &mut MemoryStore, threshold: Option<f64>) -> usize {
        let mut transferred = 0;
        for entry in &self.entries {
            let novel = match threshold {
                Some(threshold) => {
                    destination
                        .entries
                        .iter()
                        .map(|existing| destination.metric.similarity(&entry.key, &existing.key))
                        .fold(f64::NEG_INFINITY, f64::max)
                        < threshold
                }
                None => true,
            };
            if novel {
                destination.remember(entry.key.clone(), entry.value.clone());
                transferred += 1;
            }
        }
        transferred
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_evicts_oldest() {
        let mut store = MemoryStore::new("s", MemoryTier::Episodic, 2).with_capacity(2);
        store.remember(vec![1.0, 0.0], Value::Int(1));
        store.remember(vec![0.0, 1.0], Value::Int(2));
        store.remember(vec![1.0, 1.0], Value::Int(3));
        assert_eq!(store.len(), 2);
        assert_eq!(store.entries[0].value, Value::Int(2));
    }

    #[test]
    fn recall_returns_closest_first() {
        let mut store = MemoryStore::new("s", MemoryTier::Episodic, 2);
        store.remember(vec![1.0, 0.0], Value::Int(1));
        store.remember(vec![0.0, 1.0], Value::Int(2));
        let hits = store.recall(&[0.9, 0.1], 1);
        assert_eq!(hits[0].1, Value::Int(1));
    }

    #[test]
    fn remember_replaces_identical_key() {
        let mut store = MemoryStore::new("s", MemoryTier::Episodic, 2);
        store.remember(vec![1.0, 2.0], Value::Int(1));
        store.remember(vec![1.0, 2.0], Value::Int(9));
        assert_eq!(store.len(), 1);
        assert_eq!(store.entries[0].value, Value::Int(9));
    }

    #[test]
    fn consolidation_transfers_novel_entries() {
        let mut short = MemoryStore::new("short", MemoryTier::Episodic, 2);
        short.remember(vec![1.0, 0.0], Value::Int(1));
        let mut long = MemoryStore::new("long", MemoryTier::Semantic, 2);
        let moved = short.consolidate_into(&mut long, Some(0.99));
        assert_eq!(moved, 1);
        assert_eq!(long.len(), 1);
    }
}

//! Effect classification.
//!
//! An [`Effect`] names a class of observable behaviour. The domain-specific
//! instruction groups in the IR map one-to-one onto effects; everything else
//! (arithmetic, control flow, tensor shaping) is pure.

use apeiron_ir::{InstrKind, Module};
use std::collections::HashMap;

/// A class of observable effect.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Effect {
    /// Belief declaration or Bayesian update.
    Belief,
    /// Categorical distribution declaration.
    Distribution,
    /// Causal graph manipulation.
    Causal,
    /// Memory storage, consolidation, or recall.
    Memory,
    /// Plan declaration.
    Plan,
    /// Runtime architecture mutation.
    Architecture,
    /// Society construction.
    Society,
    /// Optimizer configuration.
    Optimizer,
    /// A training step.
    Training,
    /// Request routing.
    Routing,
}

impl Effect {
    /// Every effect, in a stable order.
    pub const ALL: [Effect; 10] = [
        Effect::Belief,
        Effect::Distribution,
        Effect::Causal,
        Effect::Memory,
        Effect::Plan,
        Effect::Architecture,
        Effect::Society,
        Effect::Optimizer,
        Effect::Training,
        Effect::Routing,
    ];

    /// Lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            Effect::Belief => "belief",
            Effect::Distribution => "distribution",
            Effect::Causal => "causal",
            Effect::Memory => "memory",
            Effect::Plan => "plan",
            Effect::Architecture => "architecture",
            Effect::Society => "society",
            Effect::Optimizer => "optimizer",
            Effect::Training => "training",
            Effect::Routing => "routing",
        }
    }

    /// The effect performed by an IR instruction, if any.
    pub fn of_instruction(kind: &InstrKind) -> Option<Effect> {
        match kind {
            InstrKind::Belief(_) => Some(Effect::Belief),
            InstrKind::Distribution { .. } => Some(Effect::Distribution),
            InstrKind::Causal(_) => Some(Effect::Causal),
            InstrKind::Memory(_) => Some(Effect::Memory),
            InstrKind::Plan { .. } => Some(Effect::Plan),
            InstrKind::Architecture(_) => Some(Effect::Architecture),
            InstrKind::Society(_) => Some(Effect::Society),
            InstrKind::Optimizer { .. } => Some(Effect::Optimizer),
            InstrKind::Train { .. } => Some(Effect::Training),
            InstrKind::Route { .. } => Some(Effect::Routing),
            _ => None,
        }
    }
}

impl std::fmt::Display for Effect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A compact set of [`Effect`]s.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct EffectSet(u32);

impl EffectSet {
    /// The empty (pure) set.
    pub const fn empty() -> Self {
        EffectSet(0)
    }

    fn mask(effect: Effect) -> u32 {
        1 << (effect as u32)
    }

    /// Adds an effect to the set.
    pub fn insert(&mut self, effect: Effect) {
        self.0 |= Self::mask(effect);
    }

    /// Whether the set contains an effect.
    pub fn contains(self, effect: Effect) -> bool {
        self.0 & Self::mask(effect) != 0
    }

    /// The union of two sets.
    pub fn union(self, other: EffectSet) -> EffectSet {
        EffectSet(self.0 | other.0)
    }

    /// Adds every effect of `other` in place.
    pub fn extend(&mut self, other: EffectSet) {
        self.0 |= other.0;
    }

    /// Whether no effects are present.
    pub fn is_pure(self) -> bool {
        self.0 == 0
    }

    /// Whether any effect is present.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Number of effects.
    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Iterates over the effects present, in [`Effect::ALL`] order.
    pub fn iter(self) -> impl Iterator<Item = Effect> {
        Effect::ALL
            .into_iter()
            .filter(move |effect| self.contains(*effect))
    }
}

impl std::fmt::Display for EffectSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_pure() {
            return f.write_str("pure");
        }
        let names = self.iter().map(Effect::name).collect::<Vec<_>>().join(", ");
        write!(f, "{{{names}}}")
    }
}

impl FromIterator<Effect> for EffectSet {
    fn from_iter<T: IntoIterator<Item = Effect>>(iter: T) -> Self {
        let mut set = EffectSet::empty();
        for effect in iter {
            set.insert(effect);
        }
        set
    }
}

/// Effects attributed to one function.
#[derive(Clone, Debug)]
pub struct FunctionEffects {
    /// Function name.
    pub name: String,
    /// Effects performed directly by this function's instructions.
    pub direct: EffectSet,
    /// Names of the functions this one calls.
    pub calls: Vec<String>,
    /// Direct effects plus those of every transitively called function.
    pub transitive: EffectSet,
}

impl FunctionEffects {
    /// Whether the function, including callees, is pure.
    pub fn is_pure(&self) -> bool {
        self.transitive.is_pure()
    }

    /// Whether the function itself performs no effects but calls effectful
    /// functions.
    pub fn is_indirect(&self) -> bool {
        self.direct.is_pure() && !self.transitive.is_pure()
    }
}

/// The effect summary for a whole module.
#[derive(Clone, Debug, Default)]
pub struct EffectReport {
    /// One entry per function, in module order.
    pub functions: Vec<FunctionEffects>,
    /// The union of every function's transitive effects.
    pub module_effects: EffectSet,
}

impl EffectReport {
    /// Looks up a function's effects by name.
    pub fn function(&self, name: &str) -> Option<&FunctionEffects> {
        self.functions.iter().find(|entry| entry.name == name)
    }

    /// Every function (including callees) that performs no effects.
    pub fn pure_functions(&self) -> impl Iterator<Item = &FunctionEffects> {
        self.functions.iter().filter(|entry| entry.is_pure())
    }

    /// Whether the whole module is pure.
    pub fn is_pure(&self) -> bool {
        self.module_effects.is_pure()
    }
}

impl std::fmt::Display for EffectReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "module effects: {}", self.module_effects)?;
        for entry in &self.functions {
            let tag = if entry.is_pure() {
                "pure".to_string()
            } else if entry.is_indirect() {
                format!("{} (via calls)", entry.transitive)
            } else {
                entry.transitive.to_string()
            };
            if entry.calls.is_empty() {
                writeln!(f, "  fn {}: {tag}", entry.name)?;
            } else {
                writeln!(
                    f,
                    "  fn {}: {tag} -> [{}]",
                    entry.name,
                    entry.calls.join(", ")
                )?;
            }
        }
        Ok(())
    }
}

/// Computes the effect summary for a module.
///
/// Effects are propagated along the call graph to a fixed point, so calling an
/// effectful function marks the caller as effectful too.
pub fn analyze_module(module: &Module) -> EffectReport {
    let mut entries: Vec<FunctionEffects> = module
        .functions
        .iter()
        .map(|function| {
            let mut direct = EffectSet::empty();
            let mut calls = Vec::new();
            for block in &function.blocks {
                for instruction in &block.instructions {
                    if let Some(effect) = Effect::of_instruction(&instruction.kind) {
                        direct.insert(effect);
                    }
                    if let InstrKind::Call { callee, .. } = &instruction.kind {
                        if !calls.contains(callee) {
                            calls.push(callee.clone());
                        }
                    }
                }
            }
            FunctionEffects {
                name: function.name.clone(),
                direct,
                calls,
                transitive: direct,
            }
        })
        .collect();

    let index: HashMap<String, usize> = entries
        .iter()
        .enumerate()
        .map(|(position, entry)| (entry.name.clone(), position))
        .collect();

    let mut changed = true;
    while changed {
        changed = false;
        for position in 0..entries.len() {
            let mut accumulated = entries[position].transitive;
            let calls = entries[position].calls.clone();
            for callee in &calls {
                if let Some(callee_position) = index.get(callee) {
                    accumulated.extend(entries[*callee_position].transitive);
                }
            }
            if accumulated != entries[position].transitive {
                entries[position].transitive = accumulated;
                changed = true;
            }
        }
    }

    let mut module_effects = EffectSet::empty();
    for entry in &entries {
        module_effects.extend(entry.transitive);
    }

    EffectReport {
        functions: entries,
        module_effects,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effect_set_union_and_purity() {
        let mut set = EffectSet::empty();
        assert!(set.is_pure());
        set.insert(Effect::Belief);
        set.insert(Effect::Memory);
        assert!(!set.is_pure());
        assert!(set.contains(Effect::Belief));
        assert!(!set.contains(Effect::Causal));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn classifies_instructions() {
        assert_eq!(
            Effect::of_instruction(&InstrKind::Belief(apeiron_ir::BeliefInstr::Declare {
                name: "b".into(),
                proposition: apeiron_ir::ValueId(0),
                confidence: apeiron_ir::ValueId(1),
            })),
            Some(Effect::Belief)
        );
        assert_eq!(
            Effect::of_instruction(&InstrKind::Binary {
                op: apeiron_ast::BinaryOp::Add,
                lhs: apeiron_ir::ValueId(0),
                rhs: apeiron_ir::ValueId(1),
            }),
            None
        );
    }

    #[test]
    fn effect_set_display() {
        let set: EffectSet = [Effect::Belief, Effect::Memory].into_iter().collect();
        assert_eq!(set.to_string(), "{belief, memory}");
        assert_eq!(EffectSet::empty().to_string(), "pure");
    }
}

//! Interprocedural effect analysis for Apeiron.
//!
//! The domain primitives of the language — beliefs, memory, causal graphs,
//! societies, and so on — are *effects*: they have observable consequences
//! outside pure value computation. This crate classifies IR instructions into
//! [`Effect`]s, attributes them to functions, and propagates them along the
//! call graph.
//!
//! ```
//! use apeiron_core::source::FileId;
//! use apeiron_effects::analyze_module;
//! use apeiron_ir::lower_program;
//! use apeiron_parser::parse_source;
//!
//! let program = parse_source("cause A -> B\n", FileId(0)).unwrap();
//! let module = lower_program(&program).unwrap();
//! let report = analyze_module(&module);
//! assert!(report.module_effects.contains(apeiron_effects::Effect::Causal));
//! ```

mod effect;

pub use effect::{analyze_module, Effect, EffectReport, EffectSet, FunctionEffects};

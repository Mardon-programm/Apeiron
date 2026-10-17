//! The Apeiron cognitive runtime.
//!
//! Given a lowered [`Module`](apeiron_ir::Module), the [`Evaluator`] walks its
//! SSA form and, as it runs, accumulates cognitive state:
//!
//! * [`MemoryStore`]s with embedding-keyed retrieval and consolidation;
//! * [`Belief`]s that update as evidence arrives;
//! * a [`CausalModel`] of causes, observations, interventions, and
//!   counterfactuals;
//! * [`Distribution`]s, [`Plan`]s, architecture mutations, societies, and
//!   training configuration.
//!
//! ```
//! use apeiron_core::source::FileId;
//! use apeiron_ir::lower_program;
//! use apeiron_parser::parse_source;
//! use apeiron_runtime::run_module;
//!
//! let source = "memory episodic Mem { dimension: 4 }\n";
//! let program = parse_source(source, FileId(0)).unwrap();
//! let module = lower_program(&program).unwrap();
//! let state = run_module(&module).unwrap();
//! assert!(state.memories.contains_key("Mem"));
//! ```

mod causal;
mod error;
mod eval;
#[cfg(feature = "ffi")]
mod ffi;
mod memory;
mod state;
mod value;

pub use causal::{CausalModel, Counterfactual};
pub use error::{Result, RuntimeError};
pub use eval::{run_module, run_module_with_builtins, Evaluator, NativeFn};
#[cfg(feature = "ffi")]
pub use ffi::load_externs;
pub use memory::{MemoryEntry, MemoryStore, Metric};
pub use state::{
    ArchitectureState, Belief, Distribution, OptimizerState, Plan, RuntimeState, SocietyState,
    TrainingRun,
};
pub use value::{EnumValue, Tensor, Value};

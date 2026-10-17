//! The tree-walking interpreter.
//!
//! Values flow through SSA: block parameters are simply bound when control
//! reaches a block, and every instruction writes its result into the current
//! frame. Effects mutate [`RuntimeState`], which the caller can inspect once
//! execution returns.

use std::collections::{BTreeMap, HashMap};

use apeiron_ir::{
    ArchInstr, BeliefInstr, BlockId, CausalInstr, Dim, Function, GlobalInit, InstrKind,
    Instruction, IrType, MemoryInstr, Module, SocietyInstr, Terminator, ValueId,
};

use crate::error::{Result, RuntimeError};
use crate::memory::{MemoryStore, Metric};
use crate::state::{Belief, Distribution, OptimizerState, Plan, RuntimeState, TrainingRun};
use crate::value::{self, Tensor, Value};

const MAX_CALL_DEPTH: usize = 256;

/// A native function callable from a program body.
///
/// Native functions are looked up by name when a call does not resolve to a
/// user-defined [`Function`]; the `apeiron-stdlib` crate supplies the
/// implementations for Apeiron's standard library. Externs loaded through the
/// `ffi` feature are boxed closures over a native symbol.
pub type NativeFn = Box<dyn Fn(&[Value]) -> Result<Value>>;

/// Executes a lowered module and records the cognitive state it builds.
pub struct Evaluator<'a> {
    module: &'a Module,
    functions: HashMap<&'a str, &'a Function>,
    builtins: HashMap<String, NativeFn>,
    state: RuntimeState,
    call_depth: usize,
}

impl<'a> Evaluator<'a> {
    /// Creates an evaluator for a module.
    pub fn new(module: &'a Module) -> Self {
        let functions = module
            .functions
            .iter()
            .map(|function| (function.name.as_str(), function))
            .collect();
        Self {
            module,
            functions,
            builtins: HashMap::new(),
            state: RuntimeState::new(),
            call_depth: 0,
        }
    }

    /// Creates an evaluator with a registry of native built-ins.
    pub fn with_builtins(module: &'a Module, builtins: HashMap<String, NativeFn>) -> Self {
        let mut evaluator = Self::new(module);
        evaluator.builtins = builtins;
        evaluator
    }

    /// Adds or replaces a single native built-in.
    pub fn register_builtin(&mut self, name: impl Into<String>, function: NativeFn) {
        self.builtins.insert(name.into(), function);
    }

    /// The registered native built-ins, keyed by name.
    pub fn builtins(&self) -> &HashMap<String, NativeFn> {
        &self.builtins
    }

    /// Runs the module's `main` and returns its value.
    pub fn run(&mut self) -> Result<Value> {
        let main = *self.functions.get("main").ok_or(RuntimeError::NoMain)?;
        self.execute(main, Vec::new())
    }

    /// Executes a named function with the given arguments.
    ///
    /// Falls back to the native built-in registry when `name` is not a
    /// user-defined function.
    pub fn call(&mut self, name: &str, args: Vec<Value>) -> Result<Value> {
        let function = self.functions.get(name).copied();
        if let Some(function) = function {
            return self.execute(function, args);
        }
        if let Some(builtin) = self.builtins.get(name) {
            return builtin(&args);
        }
        Err(RuntimeError::UnknownFunction {
            name: name.to_string(),
        })
    }

    /// The state built so far.
    pub fn state(&self) -> &RuntimeState {
        &self.state
    }

    /// Consumes the evaluator, yielding the final state.
    pub fn into_state(self) -> RuntimeState {
        self.state
    }

    fn execute(&mut self, function: &'a Function, args: Vec<Value>) -> Result<Value> {
        if self.call_depth >= MAX_CALL_DEPTH {
            return Err(RuntimeError::Domain("call stack overflow".into()));
        }
        self.call_depth += 1;
        let result = self.execute_inner(function, args);
        self.call_depth -= 1;
        result
    }

    fn execute_inner(&mut self, function: &'a Function, args: Vec<Value>) -> Result<Value> {
        let mut frame = Frame::new(function);
        frame.bind(function.entry, args)?;

        let mut current = function.entry;
        loop {
            let block = function.block(current);
            for instruction in &block.instructions {
                if let Some(value) = self.eval_instruction(instruction, &mut frame)? {
                    if let Some(result) = instruction.result {
                        frame.set(result, value);
                    }
                }
            }
            match &block.terminator {
                Some(Terminator::Ret(value)) => {
                    return match value {
                        Some(id) => frame.get(*id),
                        None => Ok(Value::Unit),
                    };
                }
                Some(Terminator::Jump { target, args }) => {
                    let values = frame.resolve_all(args)?;
                    frame.bind(*target, values)?;
                    current = *target;
                }
                Some(Terminator::Branch {
                    condition,
                    then_block,
                    then_args,
                    else_block,
                    else_args,
                }) => {
                    let taken = frame.get(*condition)?.is_truthy();
                    let (target, args) = if taken {
                        (then_block, then_args)
                    } else {
                        (else_block, else_args)
                    };
                    let values = frame.resolve_all(args)?;
                    frame.bind(*target, values)?;
                    current = *target;
                }
                Some(Terminator::Unreachable) => return Err(RuntimeError::Unreachable),
                None => {
                    return Err(RuntimeError::Domain(format!(
                        "block {} has no terminator",
                        block.id
                    )))
                }
            }
        }
    }

    fn eval_instruction(
        &mut self,
        instruction: &Instruction,
        frame: &mut Frame<'a>,
    ) -> Result<Option<Value>> {
        let result = match &instruction.kind {
            InstrKind::Const(constant) => Some(Value::from_const(constant)),
            InstrKind::TensorLit(literal) => Some(Value::from_tensor_lit(literal)),
            InstrKind::GlobalRef { name } => Some(self.resolve_global(name)?),
            InstrKind::MakeTensor { elements } => {
                let values = frame.resolve_all(elements)?;
                Some(value::make_tensor(&values)?)
            }
            InstrKind::Unary { op, operand } => Some(value::unary(*op, &frame.get(*operand)?)?),
            InstrKind::Binary { op, lhs, rhs } => {
                Some(value::binary(*op, &frame.get(*lhs)?, &frame.get(*rhs)?)?)
            }
            InstrKind::MatMul { lhs, rhs } => {
                Some(value::matmul(&frame.get(*lhs)?, &frame.get(*rhs)?)?)
            }
            InstrKind::Call { callee, args } => {
                let values = frame.resolve_all(args)?;
                let function = self.functions.get(callee.as_str()).copied();
                match function {
                    Some(function) => Some(self.execute(function, values)?),
                    None => match self.builtins.get(callee.as_str()) {
                        Some(builtin) => Some(builtin(&values)?),
                        None => {
                            return Err(RuntimeError::UnknownFunction {
                                name: callee.clone(),
                            })
                        }
                    },
                }
            }
            InstrKind::Index { target, index } => {
                Some(value::index(&frame.get(*target)?, &frame.get(*index)?)?)
            }
            InstrKind::Extract { tensor, index } => {
                let value = frame.get(*tensor)?;
                let Value::Tensor(tensor) = value else {
                    return Err(RuntimeError::Type("extract expects a tensor".into()));
                };
                Some(value::extract(&tensor, *index as i64)?)
            }
            InstrKind::Dimension { tensor, axis } => {
                let value = frame.get(*tensor)?;
                let Value::Tensor(tensor) = value else {
                    return Err(RuntimeError::Type("dimension expects a tensor".into()));
                };
                Some(value::dimension(&tensor, *axis)?)
            }
            InstrKind::MakeEnum {
                type_name,
                variant,
                payload,
            } => {
                let values = frame.resolve_all(payload)?;
                Some(Value::Enum(value::EnumValue::new(
                    type_name.clone(),
                    variant.clone(),
                    values,
                )))
            }
            InstrKind::EnumIs { value, variant } => {
                let Value::Enum(enum_value) = frame.get(*value)? else {
                    return Err(RuntimeError::Type("enum test expects an enum".into()));
                };
                Some(Value::Bool(enum_value.is(variant)))
            }
            InstrKind::EnumPayload { value, index } => {
                // Payload extraction is emitted alongside the variant test,
                // so a non-matching variant still runs it. Yield `unit` in
                // that case; the analyzed pattern guarantees the payload is
                // only observed once the variant matched.
                match frame.get(*value)? {
                    Value::Enum(enum_value) => Some(
                        enum_value
                            .payload
                            .get(*index as usize)
                            .cloned()
                            .unwrap_or(Value::Unit),
                    ),
                    _ => Some(Value::Unit),
                }
            }
            InstrKind::Gradient { target: _, wrt } => Some(seed_derivative(&frame.get(*wrt)?)),
            InstrKind::Belief(inner) => {
                self.eval_belief(inner, frame)?;
                None
            }
            InstrKind::Distribution { name, categories } => {
                let mut resolved = Vec::with_capacity(categories.len());
                for (category, id) in categories {
                    resolved.push((category.clone(), frame.get(*id)?.as_f64().unwrap_or(0.0)));
                }
                self.state
                    .distributions
                    .insert(name.clone(), Distribution::new(resolved));
                None
            }
            InstrKind::Causal(inner) => {
                self.eval_causal(inner, frame)?;
                None
            }
            InstrKind::Memory(inner) => {
                self.eval_memory(inner, frame)?;
                None
            }
            InstrKind::Architecture(inner) => {
                self.eval_architecture(inner, frame)?;
                None
            }
            InstrKind::Society(inner) => {
                self.eval_society(inner, frame)?;
                None
            }
            InstrKind::Plan { name, goal } => {
                let goal = frame.get(*goal)?.to_string();
                self.state.plans.insert(
                    name.clone(),
                    Plan {
                        name: name.clone(),
                        goal,
                    },
                );
                None
            }
            InstrKind::Optimizer { kind, settings } => {
                let mut resolved = BTreeMap::new();
                for (key, id) in settings {
                    resolved.insert(key.clone(), frame.get(*id)?.as_f64().unwrap_or(0.0));
                }
                self.state.optimizers.push(OptimizerState {
                    kind: kind.clone(),
                    settings: resolved,
                });
                None
            }
            InstrKind::Train { model, dataset } => {
                self.state.training.push(TrainingRun {
                    model: frame.get(*model)?.to_string(),
                    dataset: frame.get(*dataset)?.to_string(),
                });
                None
            }
            InstrKind::Route {
                request,
                cluster,
                budget,
            } => {
                let request = frame.get(*request)?.to_string();
                let mut parts = Vec::with_capacity(budget.len());
                for (key, id) in budget {
                    parts.push(format!("{key}={}", frame.get(*id)?));
                }
                self.state
                    .architecture
                    .routing
                    .push(format!("{request} -> {cluster} {{{}}}", parts.join(", ")));
                None
            }
        };
        Ok(result)
    }

    fn resolve_global(&mut self, name: &str) -> Result<Value> {
        if let Some(value) = self.state.globals.get(name) {
            return Ok(value.clone());
        }
        let value = match self.module.global(name) {
            Some(global) => match &global.init {
                Some(GlobalInit::Const(constant)) => Value::from_const(constant),
                Some(GlobalInit::Tensor(literal)) => Value::from_tensor_lit(literal),
                None => default_value(&global.ty),
            },
            None => Value::Str(name.to_string()),
        };
        self.state.globals.insert(name.to_string(), value.clone());
        Ok(value)
    }

    fn eval_belief(&mut self, inner: &BeliefInstr, frame: &Frame<'a>) -> Result<()> {
        match inner {
            BeliefInstr::Declare {
                name,
                proposition,
                confidence,
            } => {
                let proposition = frame.get(*proposition)?.to_string();
                let confidence = frame.get(*confidence)?.as_f64().unwrap_or(0.0);
                self.state
                    .beliefs
                    .insert(name.clone(), Belief::new(proposition, confidence));
            }
            BeliefInstr::Update {
                name,
                likelihood,
                reliability,
            } => {
                let likelihood = frame.get(*likelihood)?.as_f64().unwrap_or(0.0);
                let reliability = frame.get(*reliability)?.as_f64().unwrap_or(0.0);
                let belief = self
                    .state
                    .beliefs
                    .entry(name.clone())
                    .or_insert_with(|| Belief::new(name.clone(), 0.5));
                belief.update(likelihood, reliability);
            }
        }
        Ok(())
    }

    fn eval_causal(&mut self, inner: &CausalInstr, frame: &Frame<'a>) -> Result<()> {
        match inner {
            CausalInstr::Cause { from, to } => self.state.causal.add_edge(from, to),
            CausalInstr::Observe { node, value } => {
                let value = frame.get(*value)?.as_f64().unwrap_or(0.0);
                self.state.causal.observe(node, value);
            }
            CausalInstr::Intervene { node, value } => {
                let value = frame.get(*value)?.as_f64().unwrap_or(0.0);
                self.state.causal.intervene(node, value);
            }
            CausalInstr::Counterfactual { node, condition } => {
                let condition = frame.get(*condition)?.is_truthy();
                self.state.causal.record_counterfactual(node, condition);
            }
            CausalInstr::Simulate { parameter, value } => {
                let value = frame.get(*value)?.as_f64().unwrap_or(0.0);
                self.state.causal.simulate(parameter, value);
            }
        }
        Ok(())
    }

    fn eval_memory(&mut self, inner: &MemoryInstr, frame: &Frame<'a>) -> Result<()> {
        match inner {
            MemoryInstr::Declare {
                tier,
                name,
                dimension,
                capacity,
                metric,
            } => {
                let dimension = frame.get(*dimension)?.as_i64().unwrap_or(0) as usize;
                let capacity = match capacity {
                    Some(id) => frame.get(*id)?.as_i64().unwrap_or(0) as usize,
                    None => 0,
                };
                let metric = match metric {
                    Some(id) => Metric::parse(frame.get(*id)?.as_str().unwrap_or("cosine")),
                    None => Metric::Cosine,
                };
                let store = MemoryStore::new(name.clone(), *tier, dimension)
                    .with_capacity(capacity)
                    .with_metric(metric);
                self.state.memories.insert(name.clone(), store);
            }
            MemoryInstr::Remember { store, entries } => {
                let dimension = self
                    .state
                    .memories
                    .get(store)
                    .map(|store| store.dimension)
                    .unwrap_or(0);
                let mut key = Vec::new();
                let mut payload = Value::Unit;
                for (field, id) in entries {
                    let value = frame.get(*id)?;
                    match field.as_str() {
                        "key" | "query" => key = embedding(&value, dimension),
                        "value" | "payload" => payload = value,
                        _ => {}
                    }
                }
                self.state
                    .memories
                    .entry(store.clone())
                    .or_insert_with(|| MemoryStore::new(store.clone(), tier_default(), dimension))
                    .remember(key, payload);
            }
            MemoryInstr::Consolidate {
                from,
                to,
                threshold,
            } => {
                let threshold = match threshold {
                    Some(id) => Some(frame.get(*id)?.as_f64().unwrap_or(0.0)),
                    None => None,
                };
                let source = self.state.memories.get(from).cloned().ok_or_else(|| {
                    RuntimeError::Domain(format!("unknown memory store `{from}`"))
                })?;
                let dimension = source.dimension;
                let tier = source.tier;
                let destination = self
                    .state
                    .memories
                    .entry(to.clone())
                    .or_insert_with(|| MemoryStore::new(to.clone(), tier, dimension));
                source.consolidate_into(destination, threshold);
            }
            MemoryInstr::Recall { store, arguments } => {
                let Some(memory) = self.state.memories.get(store) else {
                    return Err(RuntimeError::Domain(format!(
                        "unknown memory store `{store}`"
                    )));
                };
                let dimension = memory.dimension;
                let mut query = Vec::new();
                let mut top_k = 1usize;
                for (field, id) in arguments {
                    let value = frame.get(*id)?;
                    match field.as_str() {
                        "query" | "key" => query = embedding(&value, dimension),
                        "top_k" => top_k = value.as_i64().unwrap_or(1).max(1) as usize,
                        _ => {}
                    }
                }
                self.state.last_recall = Some(memory.recall(&query, top_k));
            }
        }
        Ok(())
    }

    fn eval_architecture(&mut self, inner: &ArchInstr, frame: &Frame<'a>) -> Result<()> {
        match inner {
            ArchInstr::AddComponent { component } => {
                self.state
                    .architecture
                    .components
                    .push(frame.get(*component)?.to_string());
            }
            ArchInstr::ExpandMemory { memory, by } => {
                let amount = frame.get(*by)?.as_i64().unwrap_or(0);
                self.state
                    .architecture
                    .memory_expansions
                    .push((memory.clone(), amount));
            }
            ArchInstr::ChangeRouting { routing } => {
                self.state
                    .architecture
                    .routing
                    .push(frame.get(*routing)?.to_string());
            }
        }
        Ok(())
    }

    fn eval_society(&mut self, inner: &SocietyInstr, frame: &Frame<'a>) -> Result<()> {
        match inner {
            SocietyInstr::DeclareSociety { name } => {
                if !self.state.society.societies.contains(name) {
                    self.state.society.societies.push(name.clone());
                }
            }
            SocietyInstr::DeclareAgent {
                society,
                name,
                role,
            } => {
                if !self.state.society.societies.contains(society) {
                    self.state.society.societies.push(society.clone());
                }
                let role = frame.get(*role)?.to_string();
                self.state.society.add_agent(name.clone(), role);
            }
            SocietyInstr::DeclareCluster {
                society,
                name,
                members,
            } => {
                if !self.state.society.societies.contains(society) {
                    self.state.society.societies.push(society.clone());
                }
                self.state
                    .society
                    .add_cluster(name.clone(), members.clone());
            }
        }
        Ok(())
    }
}

fn tier_default() -> apeiron_ast::MemoryTier {
    apeiron_ast::MemoryTier::Episodic
}

/// A default value for an uninitialised global.
fn default_value(ty: &IrType) -> Value {
    match ty {
        IrType::Unit => Value::Unit,
        IrType::I64 => Value::Int(0),
        IrType::F32 | IrType::F64 => Value::Float(0.0),
        IrType::Bool => Value::Bool(false),
        IrType::Str => Value::Str(String::new()),
        IrType::Unknown => Value::Unit,
        IrType::Tensor(tensor) => {
            let shape: Vec<usize> = tensor
                .shape
                .iter()
                .map(|dim| match dim {
                    Dim::Const(extent) => *extent as usize,
                    Dim::Dynamic => 0,
                })
                .collect();
            let len = if shape.is_empty() {
                1
            } else {
                shape.iter().product()
            };
            Value::Tensor(Tensor::new(tensor.elem, shape, vec![0.0; len]))
        }
        IrType::Domain(apeiron_ir::DomainKind::CausalNode) => Value::Float(0.0),
        IrType::Enum(enum_ty) => {
            let variant = if enum_ty.name == "Result" {
                "Err"
            } else {
                "None"
            };
            Value::Enum(value::EnumValue::new(
                enum_ty.name.clone(),
                variant,
                Vec::new(),
            ))
        }
        IrType::Domain(_) => Value::Unit,
    }
}

/// Reverse-mode seeds the derivative of an objective with respect to itself
/// as one; the runtime has no autodiff, so this stands in as the identity.
fn seed_derivative(value: &Value) -> Value {
    match value {
        Value::Tensor(tensor) => Value::Tensor(Tensor::new(
            tensor.elem,
            tensor.shape.clone(),
            vec![1.0; tensor.data.len()],
        )),
        Value::Int(_) => Value::Int(1),
        Value::Float(_) => Value::Float(1.0),
        other => other.clone(),
    }
}

/// Turns any value into an embedding vector of the requested dimension.
fn embedding(value: &Value, dimension: usize) -> Vec<f64> {
    match value {
        Value::Tensor(tensor) => tensor.data.clone(),
        Value::Int(inner) => vec![*inner as f64],
        Value::Float(inner) => vec![*inner],
        Value::Str(inner) => hashed_embedding(inner, if dimension == 0 { 8 } else { dimension }),
        Value::Bool(inner) => vec![if *inner { 1.0 } else { 0.0 }],
        Value::Enum(_) => vec![0.0],
        Value::Unit => vec![0.0],
    }
}

fn hashed_embedding(text: &str, dimension: usize) -> Vec<f64> {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (0..dimension)
        .map(|_| {
            hash ^= hash << 13;
            hash ^= hash >> 7;
            hash ^= hash << 17;
            (hash as f64 / u64::MAX as f64) * 2.0 - 1.0
        })
        .collect()
}

struct Frame<'a> {
    function: &'a Function,
    values: Vec<Option<Value>>,
}

impl<'a> Frame<'a> {
    fn new(function: &'a Function) -> Self {
        Self {
            function,
            values: vec![None; function.values.len()],
        }
    }

    fn set(&mut self, id: ValueId, value: Value) {
        self.values[id.0 as usize] = Some(value);
    }

    fn get(&self, id: ValueId) -> Result<Value> {
        self.values
            .get(id.0 as usize)
            .and_then(|slot| slot.clone())
            .ok_or(RuntimeError::UndefinedValue { value: id })
    }

    fn resolve_all(&self, ids: &[ValueId]) -> Result<Vec<Value>> {
        ids.iter().map(|id| self.get(*id)).collect()
    }

    fn bind(&mut self, block: BlockId, args: Vec<Value>) -> Result<()> {
        let params = &self.function.block(block).params;
        if params.len() != args.len() {
            return Err(RuntimeError::Type(format!(
                "block {} expects {} arguments but got {}",
                block,
                params.len(),
                args.len()
            )));
        }
        for (param, value) in params.iter().zip(args) {
            self.set(*param, value);
        }
        Ok(())
    }
}

/// Lowers, then runs a module, discarding the final value.
pub fn run_module(module: &Module) -> Result<RuntimeState> {
    let mut evaluator = Evaluator::new(module);
    evaluator.run()?;
    Ok(evaluator.into_state())
}

/// Runs a module with a registry of native built-ins available to it.
pub fn run_module_with_builtins(
    module: &Module,
    builtins: HashMap<String, NativeFn>,
) -> Result<RuntimeState> {
    let mut evaluator = Evaluator::with_builtins(module, builtins);
    evaluator.run()?;
    Ok(evaluator.into_state())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashed_embeddings_are_deterministic() {
        assert_eq!(hashed_embedding("abc", 4), hashed_embedding("abc", 4));
        assert_ne!(hashed_embedding("abc", 4), hashed_embedding("abd", 4));
    }

    #[test]
    fn seed_derivative_matches_shape() {
        let tensor = Value::Tensor(Tensor::new(
            apeiron_ir::ElemType::F64,
            vec![2],
            vec![3.0, 4.0],
        ));
        let seed = seed_derivative(&tensor);
        assert_eq!(
            seed,
            Value::Tensor(Tensor::new(
                apeiron_ir::ElemType::F64,
                vec![2],
                vec![1.0, 1.0]
            ))
        );
    }

    #[test]
    fn default_tensor_is_zeroed() {
        let ty = IrType::Tensor(apeiron_ir::TensorTy::new(
            apeiron_ir::ElemType::F32,
            vec![Dim::Const(2), Dim::Const(3)],
        ));
        let value = default_value(&ty);
        assert_eq!(value.shape(), Some(&[2, 3][..]));
    }
}

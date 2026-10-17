//! Lowering from the AST into SSA IR.
//!
//! Top-level declarations become module globals; executable statements are
//! lowered into a synthetic `main` function, while each `fn` becomes its
//! own function. Control flow uses block parameters, so loop-carried
//! values and `if` results are threaded through branch arguments.

use std::collections::{BTreeSet, HashMap};
use std::fmt;

use apeiron_ast::{
    ArchOp, Arm, BeliefStmt, BinaryOp, Expr, Field, FnDecl, Ident, Item, Literal, Pattern, Program,
    SocietyMember, TypeDim, TypeExpr, UnaryOp,
};
use apeiron_core::span::Span;

use crate::builder::FunctionBuilder;
use crate::function::{Function, Global, GlobalInit, GlobalKind, Module, Terminator};
use crate::instr::{
    ArchInstr, BeliefInstr, CausalInstr, ConstValue, InstrKind, MemoryInstr, SocietyInstr,
    TensorLit,
};
use crate::types::{Dim, DomainKind, ElemType, EnumTy, IrType, TensorTy};
use crate::value::{BlockId, FunctionId, ValueId};

/// Error produced when a construct cannot be lowered.
#[derive(Clone, Debug)]
pub struct LowerError {
    /// Human readable explanation.
    pub message: String,
    /// Source range, when known.
    pub span: Option<Span>,
}

impl LowerError {
    /// Builds a lowering error.
    pub fn new(message: impl Into<String>, span: Option<Span>) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LowerError {}

/// Lowers a whole program into an IR module.
pub fn lower_program(program: &Program) -> Result<Module, LowerError> {
    Lowerer::new().run(program)
}

struct LoopContext {
    continue_block: BlockId,
    exit_block: BlockId,
    carried: Vec<String>,
}

/// A pattern test deferred to a test block.
enum MatchTest {
    /// `value == literal`.
    Literal {
        /// Value produced by lowering the scrutinee (or an extracted element).
        value: ValueId,
        /// Literal the value must equal.
        literal: Literal,
    },
    /// `value` is the named enum variant.
    Variant {
        /// Value produced by lowering the scrutinee.
        value: ValueId,
        /// Variant name to test against.
        variant: String,
    },
}

struct Lowerer {
    module: Module,
    signatures: HashMap<String, (Vec<IrType>, IrType)>,
    globals: HashMap<String, IrType>,
    env: HashMap<String, ValueId>,
    consts: HashMap<ValueId, ConstValue>,
    loops: Vec<LoopContext>,
    next_fn: u32,
}

impl Lowerer {
    fn new() -> Self {
        Self {
            module: Module::new("apeiron"),
            signatures: HashMap::new(),
            globals: HashMap::new(),
            env: HashMap::new(),
            consts: HashMap::new(),
            loops: Vec::new(),
            next_fn: 0,
        }
    }

    fn run(mut self, program: &Program) -> Result<Module, LowerError> {
        self.collect_declarations(program);
        for item in &program.items {
            if let Item::Fn(decl) = item {
                let function = self.lower_function(decl);
                self.module.functions.push(function);
            }
        }
        let main = self.lower_main(program);
        self.module.functions.push(main);
        Ok(self.module)
    }

    // ── declaration pre-pass ────────────────────────────────────────

    fn collect_declarations(&mut self, program: &Program) {
        for item in &program.items {
            match item {
                Item::Fn(decl) => {
                    let params = decl
                        .params
                        .iter()
                        .map(|param| match &param.ty {
                            Some(ty) => ir_type(ty),
                            None => IrType::F32,
                        })
                        .collect();
                    let ret = match &decl.return_type {
                        Some(ty) => ir_type(ty),
                        None => IrType::Unit,
                    };
                    self.signatures
                        .insert(decl.name.name.clone(), (params, ret));
                }
                Item::Tensor(decl) => {
                    let ty = match &decl.ty {
                        Some(ty) => ir_type(ty),
                        None => init_type(&decl.value).unwrap_or(IrType::F32),
                    };
                    let kind = match decl.kind {
                        apeiron_ast::TensorKind::Tensor => GlobalKind::Tensor,
                        apeiron_ast::TensorKind::Parameter => GlobalKind::Parameter,
                    };
                    self.add_global(
                        decl.name.name.clone(),
                        ty,
                        kind,
                        init_of(&decl.value),
                        decl.span,
                    );
                }
                Item::Memory(decl) => {
                    self.add_global(
                        decl.name.name.clone(),
                        IrType::Domain(DomainKind::Memory(decl.tier)),
                        GlobalKind::Memory,
                        None,
                        decl.span,
                    );
                }
                Item::Belief(stmt) => {
                    let name = stmt.name().name.clone();
                    self.add_global(
                        name,
                        IrType::Domain(DomainKind::Belief),
                        GlobalKind::Belief,
                        None,
                        stmt.span(),
                    );
                }
                Item::Distribution { name, span, .. } => {
                    self.add_global(
                        name.name.clone(),
                        IrType::Domain(DomainKind::Distribution),
                        GlobalKind::Distribution,
                        None,
                        *span,
                    );
                }
                Item::Plan { name, span, .. } => {
                    self.add_global(
                        name.name.clone(),
                        IrType::Domain(DomainKind::Plan),
                        GlobalKind::Plan,
                        None,
                        *span,
                    );
                }
                Item::SelfModel { name, span, .. } => {
                    self.add_global(
                        name.name.clone(),
                        IrType::Domain(DomainKind::SelfModel),
                        GlobalKind::SelfModel,
                        None,
                        *span,
                    );
                }
                Item::Architecture(block) => {
                    self.add_global(
                        block.name.name.clone(),
                        IrType::Domain(DomainKind::Architecture),
                        GlobalKind::Architecture,
                        None,
                        block.span,
                    );
                }
                Item::Society(block) => {
                    self.add_global(
                        block.name.name.clone(),
                        IrType::Domain(DomainKind::Society),
                        GlobalKind::Society,
                        None,
                        block.span,
                    );
                }
                Item::Cause { from, to, span } => {
                    for node in [from, to] {
                        self.add_global(
                            node.name.clone(),
                            IrType::Domain(DomainKind::CausalNode),
                            GlobalKind::CausalNode,
                            None,
                            *span,
                        );
                    }
                }
                _ => {}
            }
        }
    }

    fn add_global(
        &mut self,
        name: String,
        ty: IrType,
        kind: GlobalKind,
        init: Option<GlobalInit>,
        span: Span,
    ) {
        if self.globals.contains_key(&name) {
            return;
        }
        self.globals.insert(name.clone(), ty.clone());
        let mut global = Global::new(name, ty, kind).with_span(span);
        if let Some(init) = init {
            global = global.with_init(init);
        }
        self.module.globals.push(global);
    }

    // ── functions ───────────────────────────────────────────────────

    fn lower_function(&mut self, decl: &FnDecl) -> Function {
        let (params, ret) = self
            .signatures
            .get(&decl.name.name)
            .cloned()
            .unwrap_or_else(|| (Vec::new(), IrType::Unit));
        self.reset_function_state();
        let mut builder =
            FunctionBuilder::new(decl.name.name.clone(), &params, ret.clone()).with_span(decl.span);
        for (index, param) in decl.params.iter().enumerate() {
            let value = builder.param(index);
            self.env.insert(param.name.name.clone(), value);
        }
        let body = self.lower_expr(&mut builder, &decl.body);
        if !builder.is_terminated() {
            if ret == IrType::Unit {
                builder.terminate(Terminator::Ret(None));
            } else {
                builder.terminate(Terminator::Ret(Some(body)));
            }
        }
        let mut function = builder.build();
        function.id = FunctionId(self.next_fn);
        self.next_fn += 1;
        function
    }

    fn lower_main(&mut self, program: &Program) -> Function {
        self.reset_function_state();
        let mut builder = FunctionBuilder::new("main", &[], IrType::Unit);
        for item in &program.items {
            if matches!(item, Item::Fn(_)) {
                continue;
            }
            self.lower_item(&mut builder, item);
        }
        if !builder.is_terminated() {
            builder.terminate(Terminator::Ret(None));
        }
        let mut function = builder.build();
        function.id = FunctionId(self.next_fn);
        self.next_fn += 1;
        function
    }

    fn reset_function_state(&mut self) {
        self.env.clear();
        self.consts.clear();
        self.loops.clear();
    }

    // ── items ───────────────────────────────────────────────────────

    fn lower_item(&mut self, builder: &mut FunctionBuilder, item: &Item) -> Option<ValueId> {
        match item {
            Item::Fn(_) => None,
            Item::Let(decl) => {
                let value = self.lower_expr(builder, &decl.value);
                self.env.insert(decl.name.name.clone(), value);
                None
            }
            Item::Tensor(decl) => {
                let value = self.lower_expr(builder, &decl.value);
                self.env.insert(decl.name.name.clone(), value);
                None
            }
            Item::Return { value, .. } => match value {
                Some(value) => {
                    let lowered = self.lower_expr(builder, value);
                    builder.terminate(Terminator::Ret(Some(lowered)));
                    Some(lowered)
                }
                None => {
                    let unit = builder.emit_const(ConstValue::Unit, IrType::Unit);
                    builder.terminate(Terminator::Ret(None));
                    Some(unit)
                }
            },
            Item::While {
                condition, body, ..
            } => {
                self.lower_while(builder, condition, body);
                None
            }
            Item::For {
                variable,
                iterable,
                body,
                ..
            } => {
                self.lower_for(builder, variable, iterable, body);
                None
            }
            Item::Break { .. } => {
                self.lower_break(builder);
                None
            }
            Item::Continue { .. } => {
                self.lower_continue(builder);
                None
            }
            Item::Expr(expr) => {
                let value = self.lower_expr(builder, expr);
                Some(value)
            }
            Item::Optimizer {
                kind,
                settings,
                span,
            } => {
                let settings = self.lower_fields(builder, settings);
                builder.emit_effect(
                    InstrKind::Optimizer {
                        kind: kind.name.clone(),
                        settings,
                    },
                    Some(*span),
                );
                None
            }
            Item::Train {
                model,
                dataset,
                span,
            } => {
                let model = self.lower_name(builder, model);
                let dataset = self.lower_name(builder, dataset);
                builder.emit_effect(InstrKind::Train { model, dataset }, Some(*span));
                None
            }
            Item::Gradient {
                target,
                with_respect_to,
                span,
            } => {
                let target = self.lower_name(builder, target);
                let wrt = self.lower_name(builder, with_respect_to);
                let ty = builder.value_type(wrt).clone();
                builder.emit(InstrKind::Gradient { target, wrt }, ty, Some(*span));
                None
            }
            Item::Distribution {
                name,
                categories,
                span,
            } => {
                let categories = self.lower_fields(builder, categories);
                let value = builder.emit(
                    InstrKind::Distribution {
                        name: name.name.clone(),
                        categories,
                    },
                    IrType::Domain(DomainKind::Distribution),
                    Some(*span),
                );
                self.env.insert(name.name.clone(), value);
                None
            }
            Item::Belief(stmt) => {
                self.lower_belief(builder, stmt);
                None
            }
            Item::Cause { from, to, span } => {
                builder.emit_effect(
                    InstrKind::Causal(CausalInstr::Cause {
                        from: from.name.clone(),
                        to: to.name.clone(),
                    }),
                    Some(*span),
                );
                None
            }
            Item::Observe {
                target,
                value,
                span,
            } => {
                let value = self.lower_expr(builder, value);
                builder.emit_effect(
                    InstrKind::Causal(CausalInstr::Observe {
                        node: target.name.clone(),
                        value,
                    }),
                    Some(*span),
                );
                None
            }
            Item::Intervene {
                target,
                value,
                span,
            } => {
                let value = self.lower_expr(builder, value);
                builder.emit_effect(
                    InstrKind::Causal(CausalInstr::Intervene {
                        node: target.name.clone(),
                        value,
                    }),
                    Some(*span),
                );
                None
            }
            Item::Counterfactual {
                target,
                condition,
                span,
            } => {
                let condition = self.lower_expr(builder, condition);
                builder.emit_effect(
                    InstrKind::Causal(CausalInstr::Counterfactual {
                        node: target.name.clone(),
                        condition,
                    }),
                    Some(*span),
                );
                None
            }
            Item::Simulate {
                parameter,
                value,
                span,
            } => {
                let value = self.lower_expr(builder, value);
                builder.emit_effect(
                    InstrKind::Causal(CausalInstr::Simulate {
                        parameter: parameter.name.clone(),
                        value,
                    }),
                    Some(*span),
                );
                None
            }
            Item::Memory(decl) => {
                let dimension = decl
                    .fields
                    .iter()
                    .find(|field| field.name.name == "dimension")
                    .map(|field| self.lower_expr(builder, &field.value))
                    .unwrap_or_else(|| builder.const_i64(0));
                let capacity = decl
                    .fields
                    .iter()
                    .find(|field| field.name.name == "capacity")
                    .map(|field| self.lower_expr(builder, &field.value));
                let metric = decl
                    .fields
                    .iter()
                    .find(|field| field.name.name == "metric")
                    .map(|field| self.lower_expr(builder, &field.value));
                let value = builder.emit(
                    InstrKind::Memory(MemoryInstr::Declare {
                        tier: decl.tier,
                        name: decl.name.name.clone(),
                        dimension,
                        capacity,
                        metric,
                    }),
                    IrType::Domain(DomainKind::Memory(decl.tier)),
                    Some(decl.span),
                );
                self.env.insert(decl.name.name.clone(), value);
                None
            }
            Item::Remember {
                memory,
                entries,
                span,
            } => {
                let entries = self.lower_fields(builder, entries);
                builder.emit_effect(
                    InstrKind::Memory(MemoryInstr::Remember {
                        store: memory.name.clone(),
                        entries,
                    }),
                    Some(*span),
                );
                None
            }
            Item::Consolidate {
                from,
                to,
                threshold,
                span,
            } => {
                let threshold = threshold
                    .as_ref()
                    .map(|value| self.lower_expr(builder, value));
                builder.emit_effect(
                    InstrKind::Memory(MemoryInstr::Consolidate {
                        from: from.name.clone(),
                        to: to.name.clone(),
                        threshold,
                    }),
                    Some(*span),
                );
                None
            }
            Item::Recall {
                memory,
                arguments,
                span,
            } => {
                let arguments = self.lower_fields(builder, arguments);
                builder.emit_effect(
                    InstrKind::Memory(MemoryInstr::Recall {
                        store: memory.name.clone(),
                        arguments,
                    }),
                    Some(*span),
                );
                None
            }
            Item::Plan { name, body, span } => {
                let goal = body
                    .iter()
                    .find(|field| field.name.name == "goal")
                    .map(|field| self.lower_expr(builder, &field.value))
                    .unwrap_or_else(|| builder.const_str(""));
                let value = builder.emit(
                    InstrKind::Plan {
                        name: name.name.clone(),
                        goal,
                    },
                    IrType::Domain(DomainKind::Plan),
                    Some(*span),
                );
                self.env.insert(name.name.clone(), value);
                None
            }
            Item::SelfModel { name, span, .. } => {
                let value = builder.emit(
                    InstrKind::GlobalRef {
                        name: name.name.clone(),
                    },
                    IrType::Domain(DomainKind::SelfModel),
                    Some(*span),
                );
                self.env.insert(name.name.clone(), value);
                None
            }
            Item::Architecture(block) => {
                for operation in &block.operations {
                    self.lower_arch_op(builder, operation);
                }
                let value = builder.emit(
                    InstrKind::GlobalRef {
                        name: block.name.name.clone(),
                    },
                    IrType::Domain(DomainKind::Architecture),
                    Some(block.span),
                );
                self.env.insert(block.name.name.clone(), value);
                None
            }
            Item::Society(block) => {
                let society = block.name.name.clone();
                let value = builder.emit(
                    InstrKind::Society(SocietyInstr::DeclareSociety {
                        name: society.clone(),
                    }),
                    IrType::Domain(DomainKind::Society),
                    Some(block.span),
                );
                self.env.insert(society.clone(), value);
                for member in &block.members {
                    match member {
                        SocietyMember::Agent { name, fields, span } => {
                            let role = fields
                                .iter()
                                .find(|field| field.name.name == "role")
                                .map(|field| self.lower_expr(builder, &field.value))
                                .unwrap_or_else(|| builder.const_str(""));
                            builder.emit_effect(
                                InstrKind::Society(SocietyInstr::DeclareAgent {
                                    society: society.clone(),
                                    name: name.name.clone(),
                                    role,
                                }),
                                Some(*span),
                            );
                        }
                        SocietyMember::Cluster {
                            name,
                            members,
                            span,
                        } => {
                            builder.emit_effect(
                                InstrKind::Society(SocietyInstr::DeclareCluster {
                                    society: society.clone(),
                                    name: name.name.clone(),
                                    members: members.iter().map(|m| m.name.clone()).collect(),
                                }),
                                Some(*span),
                            );
                        }
                    }
                }
                None
            }
            Item::Route {
                request,
                cluster,
                budget,
                span,
            } => {
                let request = self.lower_name(builder, request);
                let budget = self.lower_fields(builder, budget);
                builder.emit_effect(
                    InstrKind::Route {
                        request,
                        cluster: cluster.name.clone(),
                        budget,
                    },
                    Some(*span),
                );
                None
            }
        }
    }

    fn lower_belief(&mut self, builder: &mut FunctionBuilder, stmt: &BeliefStmt) {
        match stmt {
            BeliefStmt::Declare { name, fields, span } => {
                let proposition = field_value(self, builder, fields, "proposition")
                    .unwrap_or_else(|| builder.const_str(""));
                let confidence = field_value(self, builder, fields, "confidence")
                    .unwrap_or_else(|| builder.const_float(0.0, ElemType::F32));
                let value = builder.emit(
                    InstrKind::Belief(BeliefInstr::Declare {
                        name: name.name.clone(),
                        proposition,
                        confidence,
                    }),
                    IrType::Domain(DomainKind::Belief),
                    Some(*span),
                );
                self.env.insert(name.name.clone(), value);
            }
            BeliefStmt::Update { name, fields, span } => {
                let likelihood = field_value(self, builder, fields, "likelihood")
                    .unwrap_or_else(|| builder.const_float(0.0, ElemType::F32));
                let reliability = field_value(self, builder, fields, "reliability")
                    .unwrap_or_else(|| builder.const_float(0.0, ElemType::F32));
                builder.emit_effect(
                    InstrKind::Belief(BeliefInstr::Update {
                        name: name.name.clone(),
                        likelihood,
                        reliability,
                    }),
                    Some(*span),
                );
            }
        }
    }

    fn lower_arch_op(&mut self, builder: &mut FunctionBuilder, operation: &ArchOp) {
        match operation {
            ArchOp::AddComponent { component, span } => {
                let component = self.lower_expr(builder, component);
                builder.emit_effect(
                    InstrKind::Architecture(ArchInstr::AddComponent { component }),
                    Some(*span),
                );
            }
            ArchOp::ExpandMemory { memory, by, span } => {
                let name = memory.as_str_literal().unwrap_or_default().to_string();
                let by = self.lower_expr(builder, by);
                builder.emit_effect(
                    InstrKind::Architecture(ArchInstr::ExpandMemory { memory: name, by }),
                    Some(*span),
                );
            }
            ArchOp::ChangeRouting { routing, span } => {
                let routing = self.lower_expr(builder, routing);
                builder.emit_effect(
                    InstrKind::Architecture(ArchInstr::ChangeRouting { routing }),
                    Some(*span),
                );
            }
        }
    }

    fn lower_fields(
        &mut self,
        builder: &mut FunctionBuilder,
        fields: &[Field],
    ) -> Vec<(String, ValueId)> {
        fields
            .iter()
            .map(|field| {
                (
                    field.name.name.clone(),
                    self.lower_expr(builder, &field.value),
                )
            })
            .collect()
    }

    // ── control flow ────────────────────────────────────────────────

    fn lower_while(&mut self, builder: &mut FunctionBuilder, condition: &Expr, body: &Expr) {
        let carried = self.carried_variables(body);
        let param_types: Vec<IrType> = carried
            .iter()
            .map(|name| self.env_type(builder, name))
            .collect();
        let header = builder.create_block_with_params("while.header", &param_types);
        let body_block = builder.create_block("while.body");
        let exit = builder.create_block_with_params("while.exit", &param_types);

        let init: Vec<ValueId> = carried.iter().map(|name| self.env_value(name)).collect();
        builder.terminate(Terminator::Jump {
            target: header,
            args: init,
        });

        builder.switch_to(header);
        for (index, name) in carried.iter().enumerate() {
            let param = builder.block_params(header)[index];
            self.env.insert(name.clone(), param);
        }
        let cond = self.lower_expr(builder, condition);
        builder.terminate(Terminator::Branch {
            condition: cond,
            then_block: body_block,
            then_args: Vec::new(),
            else_block: exit,
            else_args: carried.iter().map(|name| self.env_value(name)).collect(),
        });

        self.loops.push(LoopContext {
            continue_block: header,
            exit_block: exit,
            carried: carried.clone(),
        });
        builder.switch_to(body_block);
        self.lower_expr(builder, body);
        let back: Vec<ValueId> = carried.iter().map(|name| self.env_value(name)).collect();
        if !builder.is_terminated() {
            builder.terminate(Terminator::Jump {
                target: header,
                args: back,
            });
        }
        self.loops.pop();

        builder.switch_to(exit);
        for (index, name) in carried.iter().enumerate() {
            let param = builder.block_params(exit)[index];
            self.env.insert(name.clone(), param);
        }
    }

    fn lower_for(
        &mut self,
        builder: &mut FunctionBuilder,
        variable: &Ident,
        iterable: &Expr,
        body: &Expr,
    ) {
        let iterable = self.lower_expr(builder, iterable);
        let length = builder.emit(
            InstrKind::Dimension {
                tensor: iterable,
                axis: 0,
            },
            IrType::I64,
            None,
        );
        let zero = builder.const_i64(0);

        let carried = self.carried_variables(body);
        let carried_types: Vec<IrType> = carried
            .iter()
            .map(|name| self.env_type(builder, name))
            .collect();

        let mut header_params = vec![IrType::I64];
        header_params.extend(carried_types.iter().cloned());
        let header = builder.create_block_with_params("for.header", &header_params);
        let body_block = builder.create_block("for.body");
        let step = builder.create_block("for.step");
        let exit = builder.create_block_with_params("for.exit", &carried_types);

        let mut init = vec![zero];
        init.extend(carried.iter().map(|name| self.env_value(name)));
        builder.terminate(Terminator::Jump {
            target: header,
            args: init,
        });

        builder.switch_to(header);
        let index = builder.block_params(header)[0];
        for (offset, name) in carried.iter().enumerate() {
            let param = builder.block_params(header)[offset + 1];
            self.env.insert(name.clone(), param);
        }
        let less = builder.emit(
            InstrKind::Binary {
                op: BinaryOp::Lt,
                lhs: index,
                rhs: length,
            },
            IrType::Bool,
            None,
        );
        builder.terminate(Terminator::Branch {
            condition: less,
            then_block: body_block,
            then_args: Vec::new(),
            else_block: exit,
            else_args: carried.iter().map(|name| self.env_value(name)).collect(),
        });

        builder.switch_to(body_block);
        let element = builder.emit(
            InstrKind::Index {
                target: iterable,
                index,
            },
            element_type(builder.value_type(iterable)),
            None,
        );
        self.env.insert(variable.name.clone(), element);
        self.loops.push(LoopContext {
            continue_block: step,
            exit_block: exit,
            carried: carried.clone(),
        });
        self.lower_expr(builder, body);
        if !builder.is_terminated() {
            builder.terminate(Terminator::Jump {
                target: step,
                args: Vec::new(),
            });
        }
        self.loops.pop();
        self.env.remove(&variable.name);

        builder.switch_to(step);
        let one = builder.const_i64(1);
        let next = builder.emit(
            InstrKind::Binary {
                op: BinaryOp::Add,
                lhs: index,
                rhs: one,
            },
            IrType::I64,
            None,
        );
        let mut back = vec![next];
        back.extend(carried.iter().map(|name| self.env_value(name)));
        builder.terminate(Terminator::Jump {
            target: header,
            args: back,
        });

        builder.switch_to(exit);
        for (offset, name) in carried.iter().enumerate() {
            let param = builder.block_params(exit)[offset];
            self.env.insert(name.clone(), param);
        }
    }

    fn lower_break(&mut self, builder: &mut FunctionBuilder) {
        if let Some(context) = self.loops.last() {
            let args: Vec<ValueId> = context
                .carried
                .iter()
                .map(|name| self.env_value(name))
                .collect();
            let exit = context.exit_block;
            builder.terminate(Terminator::Jump { target: exit, args });
        }
    }

    fn lower_continue(&mut self, builder: &mut FunctionBuilder) {
        if let Some(context) = self.loops.last() {
            let args: Vec<ValueId> = context
                .carried
                .iter()
                .map(|name| self.env_value(name))
                .collect();
            let target = context.continue_block;
            builder.terminate(Terminator::Jump { target, args });
        }
    }

    // ── expressions ─────────────────────────────────────────────────

    fn lower_expr(&mut self, builder: &mut FunctionBuilder, expr: &Expr) -> ValueId {
        match expr {
            Expr::Literal { value, .. } => self.lower_literal(builder, value),
            Expr::Ident(ident) => self.lower_name(builder, ident),
            Expr::Array { elements, .. } => self.lower_array(builder, elements),
            Expr::Unary { op, operand, span } => {
                let operand = self.lower_expr(builder, operand);
                let ty = match op {
                    UnaryOp::Neg => builder.value_type(operand).clone(),
                    UnaryOp::Not => IrType::Bool,
                };
                builder.emit(InstrKind::Unary { op: *op, operand }, ty, Some(*span))
            }
            Expr::Binary { op, lhs, rhs, span } => self.lower_binary(builder, *op, lhs, rhs, *span),
            Expr::Call { callee, args, span } => self.lower_call(builder, callee, args, *span),
            Expr::Index {
                target,
                index,
                span,
            } => {
                let target = self.lower_expr(builder, target);
                let index = self.lower_expr(builder, index);
                let ty = element_type(builder.value_type(target));
                builder.emit(InstrKind::Index { target, index }, ty, Some(*span))
            }
            Expr::Field { target, .. } => self.lower_expr(builder, target),
            Expr::If {
                condition,
                then_branch,
                else_branch,
                span,
            } => self.lower_if(
                builder,
                condition,
                then_branch,
                else_branch.as_deref(),
                *span,
            ),
            Expr::Block { items, .. } => {
                let mut last = None;
                for item in items {
                    last = self.lower_item(builder, item);
                    if builder.is_terminated() {
                        break;
                    }
                }
                match last {
                    Some(value) => value,
                    None if builder.is_terminated() => placeholder_value(builder),
                    None => builder.emit_const(ConstValue::Unit, IrType::Unit),
                }
            }
            Expr::Match { expr, arms, span } => self.lower_match(builder, expr, arms, *span),
            Expr::Try { operand, span } => self.lower_try(builder, operand, *span),
        }
    }

    fn lower_literal(&mut self, builder: &mut FunctionBuilder, literal: &Literal) -> ValueId {
        match literal {
            Literal::Int(value) => builder.const_i64(*value),
            Literal::Float(value) => builder.const_float(*value, ElemType::F32),
            Literal::Bool(value) => builder.emit_const(ConstValue::Bool(*value), IrType::Bool),
            Literal::Str(value) => builder.const_str(value.clone()),
        }
    }

    fn lower_name(&mut self, builder: &mut FunctionBuilder, ident: &Ident) -> ValueId {
        if let Some(value) = self.env.get(&ident.name) {
            return *value;
        }
        if ident.name == "None" {
            return builder.emit(
                InstrKind::MakeEnum {
                    type_name: "Option".to_string(),
                    variant: "None".to_string(),
                    payload: Vec::new(),
                },
                IrType::Enum(EnumTy::new("Option", vec![IrType::Unknown])),
                Some(ident.span),
            );
        }
        let ty = self
            .globals
            .get(&ident.name)
            .cloned()
            .unwrap_or(IrType::Str);
        builder.emit(
            InstrKind::GlobalRef {
                name: ident.name.clone(),
            },
            ty,
            Some(ident.span),
        )
    }

    fn lower_array(&mut self, builder: &mut FunctionBuilder, elements: &[Expr]) -> ValueId {
        if let Some(literal) = tensor_literal(elements) {
            let ty = IrType::Tensor(TensorTy::new(
                literal.elem,
                literal
                    .shape
                    .iter()
                    .map(|extent| Dim::Const(*extent))
                    .collect(),
            ));
            return builder.emit(InstrKind::TensorLit(literal), ty, None);
        }
        let lowered: Vec<ValueId> = elements
            .iter()
            .map(|element| self.lower_expr(builder, element))
            .collect();
        let ty = make_tensor_type(builder, &lowered);
        builder.emit(InstrKind::MakeTensor { elements: lowered }, ty, None)
    }

    fn lower_binary(
        &mut self,
        builder: &mut FunctionBuilder,
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
        span: Span,
    ) -> ValueId {
        if op == BinaryOp::Assign {
            let value = self.lower_expr(builder, rhs);
            if let Expr::Ident(ident) = lhs {
                self.env.insert(ident.name.clone(), value);
            } else {
                let _ = self.lower_expr(builder, lhs);
            }
            return value;
        }
        let lhs_value = self.lower_expr(builder, lhs);
        let rhs_value = self.lower_expr(builder, rhs);
        let lhs_ty = builder.value_type(lhs_value).clone();
        let rhs_ty = builder.value_type(rhs_value).clone();
        if op == BinaryOp::MatMul {
            let ty = matmul_type(&lhs_ty, &rhs_ty);
            return builder.emit(
                InstrKind::MatMul {
                    lhs: lhs_value,
                    rhs: rhs_value,
                },
                ty,
                Some(span),
            );
        }
        let ty = if op.is_comparison() {
            IrType::Bool
        } else if lhs_ty.is_tensor() {
            lhs_ty
        } else if rhs_ty.is_tensor() {
            rhs_ty
        } else {
            lhs_ty
        };
        builder.emit(
            InstrKind::Binary {
                op,
                lhs: lhs_value,
                rhs: rhs_value,
            },
            ty,
            Some(span),
        )
    }

    fn lower_call(
        &mut self,
        builder: &mut FunctionBuilder,
        callee: &Expr,
        args: &[Expr],
        span: Span,
    ) -> ValueId {
        let name = match callee {
            Expr::Ident(ident) => ident.name.clone(),
            _ => "<expr>".to_string(),
        };
        let lowered: Vec<ValueId> = args
            .iter()
            .map(|arg| self.lower_expr(builder, arg))
            .collect();
        if matches!(name.as_str(), "Some" | "Ok" | "Err") {
            let payload_ty = lowered
                .first()
                .map(|value| builder.value_type(*value).clone())
                .unwrap_or(IrType::F32);
            let (type_name, payload) = match name.as_str() {
                "Some" => ("Option", vec![payload_ty]),
                "Ok" => ("Result", vec![payload_ty, IrType::Unknown]),
                _ => ("Result", vec![IrType::Unknown, payload_ty]),
            };
            return builder.emit(
                InstrKind::MakeEnum {
                    type_name: type_name.to_string(),
                    variant: name,
                    payload: lowered,
                },
                IrType::Enum(EnumTy::new(type_name, payload)),
                Some(span),
            );
        }
        let ty = self
            .signatures
            .get(&name)
            .map(|(_, ret)| ret.clone())
            .unwrap_or_else(|| builtin_ret_type(&name).unwrap_or(IrType::Unit));
        builder.emit(
            InstrKind::Call {
                callee: name,
                args: lowered,
            },
            ty,
            Some(span),
        )
    }

    fn lower_if(
        &mut self,
        builder: &mut FunctionBuilder,
        condition: &Expr,
        then_branch: &Expr,
        else_branch: Option<&Expr>,
        _span: Span,
    ) -> ValueId {
        let cond = self.lower_expr(builder, condition);
        let then_block = builder.create_block("if.then");
        let else_block = builder.create_block("if.else");
        builder.terminate(Terminator::Branch {
            condition: cond,
            then_block,
            then_args: Vec::new(),
            else_block,
            else_args: Vec::new(),
        });

        builder.switch_to(then_block);
        let then_value = self.lower_expr(builder, then_branch);
        let then_terminated = builder.is_terminated();
        let then_ty = builder.value_type(then_value).clone();

        let else_value = match else_branch {
            Some(else_branch) => {
                builder.switch_to(else_block);
                let value = self.lower_expr(builder, else_branch);
                Some((value, builder.is_terminated()))
            }
            None => None,
        };

        match else_value {
            Some((else_value, else_terminated)) if !then_terminated || !else_terminated => {
                let else_ty = builder.value_type(else_value).clone();
                let common = common_type(&then_ty, &else_ty);
                let merge = builder.create_block_with_params("if.merge", &[common]);
                let param = builder.block_params(merge)[0];
                if !then_terminated {
                    builder.switch_to(then_block);
                    builder.terminate(Terminator::Jump {
                        target: merge,
                        args: vec![then_value],
                    });
                }
                if !else_terminated {
                    builder.switch_to(else_block);
                    builder.terminate(Terminator::Jump {
                        target: merge,
                        args: vec![else_value],
                    });
                }
                builder.switch_to(merge);
                param
            }
            _ => {
                let merge = builder.create_block("if.merge");
                if !then_terminated {
                    builder.switch_to(then_block);
                    builder.terminate(Terminator::Jump {
                        target: merge,
                        args: Vec::new(),
                    });
                }
                if let Some((_, else_terminated)) = else_value {
                    if !else_terminated {
                        builder.switch_to(else_block);
                        builder.terminate(Terminator::Jump {
                            target: merge,
                            args: Vec::new(),
                        });
                    }
                } else {
                    builder.switch_to(else_block);
                    builder.terminate(Terminator::Jump {
                        target: merge,
                        args: Vec::new(),
                    });
                }
                builder.switch_to(merge);
                builder.emit_const(ConstValue::Unit, IrType::Unit)
            }
        }
    }

    // ── match ───────────────────────────────────────────────────────

    /// Lowers `operand?`: returns early on the failure variant, otherwise
    /// unwraps the success payload.
    fn lower_try(&mut self, builder: &mut FunctionBuilder, operand: &Expr, span: Span) -> ValueId {
        let value = self.lower_expr(builder, operand);
        let enum_ty = match builder.value_type(value) {
            IrType::Enum(enum_ty) => enum_ty.clone(),
            _ => EnumTy::new("Option", Vec::new()),
        };
        let bad_variant = if enum_ty.name == "Result" {
            "Err"
        } else {
            "None"
        };
        let is_bad = builder.emit(
            InstrKind::EnumIs {
                value,
                variant: bad_variant.to_string(),
            },
            IrType::Bool,
            Some(span),
        );
        let propagate = builder.create_block("try.propagate");
        let ok_block = builder.create_block("try.ok");
        builder.terminate(Terminator::Branch {
            condition: is_bad,
            then_block: propagate,
            then_args: Vec::new(),
            else_block: ok_block,
            else_args: Vec::new(),
        });
        builder.switch_to(propagate);
        builder.terminate(Terminator::Ret(Some(value)));
        builder.switch_to(ok_block);
        let payload_ty = enum_ty.payload_at(0).cloned().unwrap_or(IrType::Unknown);
        builder.emit(
            InstrKind::EnumPayload { value, index: 0 },
            payload_ty,
            Some(span),
        )
    }

    /// Lowers a `match` expression into a chain of test blocks feeding the
    /// arm bodies; arm results merge through a block parameter.
    fn lower_match(
        &mut self,
        builder: &mut FunctionBuilder,
        scrutinee: &Expr,
        arms: &[Arm],
        _span: Span,
    ) -> ValueId {
        let value = self.lower_expr(builder, scrutinee);
        let outer_env = self.env.clone();

        let fail_block = builder.create_block("match.fail");
        let arm_blocks: Vec<BlockId> = (0..arms.len())
            .map(|index| builder.create_block(format!("match.arm.{index}")))
            .collect();
        let entry_blocks: Vec<BlockId> = (0..arms.len())
            .map(|index| builder.create_block(format!("match.test.{index}")))
            .collect();

        let first = entry_blocks.first().copied().unwrap_or(fail_block);
        builder.terminate(Terminator::Jump {
            target: first,
            args: Vec::new(),
        });

        // Emit the pattern tests for every arm, remembering the bindings
        // each pattern introduces so the arm bodies can see them.
        let mut arm_bindings: Vec<Vec<(String, ValueId)>> = Vec::with_capacity(arms.len());
        for (index, arm) in arms.iter().enumerate() {
            builder.switch_to(entry_blocks[index]);
            self.env = outer_env.clone();

            let mut bindings = Vec::new();
            let mut tests = Vec::new();
            self.lower_pattern_tests(
                builder,
                &arm.pat,
                value,
                &mut bindings,
                &mut tests,
                arm.span,
            );
            for (name, bound) in &bindings {
                self.env.insert(name.clone(), *bound);
            }
            arm_bindings.push(bindings);

            let fail_target = entry_blocks.get(index + 1).copied().unwrap_or(fail_block);
            let total = tests.len() + usize::from(arm.guard.is_some());
            let mut step = 0;
            for test in &tests {
                step += 1;
                let cond = self.lower_test(builder, test, arm.span);
                let last = step == total;
                let then_block = if last {
                    arm_blocks[index]
                } else {
                    builder.create_block(format!("match.test.{index}.{step}"))
                };
                builder.terminate(Terminator::Branch {
                    condition: cond,
                    then_block,
                    then_args: Vec::new(),
                    else_block: fail_target,
                    else_args: Vec::new(),
                });
                if !last {
                    builder.switch_to(then_block);
                }
            }
            if let Some(guard) = &arm.guard {
                let cond = self.lower_expr(builder, guard);
                builder.terminate(Terminator::Branch {
                    condition: cond,
                    then_block: arm_blocks[index],
                    then_args: Vec::new(),
                    else_block: fail_target,
                    else_args: Vec::new(),
                });
            } else if total == 0 {
                builder.terminate(Terminator::Jump {
                    target: arm_blocks[index],
                    args: Vec::new(),
                });
            }
        }

        self.env = outer_env.clone();
        builder.switch_to(fail_block);
        builder.terminate(Terminator::Unreachable);

        // Lower each arm body and remember the block and value it ends in.
        let mut arm_values: Vec<Option<(BlockId, ValueId)>> = Vec::with_capacity(arms.len());
        let mut common: Option<IrType> = None;
        for (index, arm) in arms.iter().enumerate() {
            builder.switch_to(arm_blocks[index]);
            self.env = outer_env.clone();
            for (name, bound) in &arm_bindings[index] {
                self.env.insert(name.clone(), *bound);
            }
            let body = self.lower_expr(builder, &arm.body);
            if builder.is_terminated() {
                arm_values.push(None);
            } else {
                let block = builder.current_block();
                let ty = builder.value_type(body).clone();
                common = Some(match common {
                    Some(previous) => common_type(&previous, &ty),
                    None => ty,
                });
                arm_values.push(Some((block, body)));
            }
        }

        match common {
            Some(common) => {
                let merge = builder.create_block_with_params("match.merge", &[common]);
                let param = builder.block_params(merge)[0];
                for entry in arm_values.iter().flatten() {
                    let (block, value) = *entry;
                    builder.switch_to(block);
                    builder.terminate(Terminator::Jump {
                        target: merge,
                        args: vec![value],
                    });
                }
                builder.switch_to(merge);
                param
            }
            None => {
                let merge = builder.create_block("match.merge");
                builder.switch_to(merge);
                builder.emit_const(ConstValue::Unit, IrType::Unit)
            }
        }
    }

    /// Emits the extraction and equality tests for one pattern against the
    /// already-lowered `value`, recording any names the pattern binds.
    fn lower_pattern_tests(
        &mut self,
        builder: &mut FunctionBuilder,
        pat: &Pattern,
        value: ValueId,
        bindings: &mut Vec<(String, ValueId)>,
        tests: &mut Vec<MatchTest>,
        span: Span,
    ) {
        match pat {
            Pattern::Wildcard => {}
            Pattern::Bind(ident) => bindings.push((ident.name.clone(), value)),
            Pattern::Literal(literal) => tests.push(MatchTest::Literal {
                value,
                literal: literal.clone(),
            }),
            Pattern::Variant { name, inner } => {
                tests.push(MatchTest::Variant {
                    value,
                    variant: name.name.clone(),
                });
                if let Some(inner) = inner {
                    let payload_ty = match builder.value_type(value) {
                        IrType::Enum(enum_ty) => {
                            enum_ty.payload_at(0).cloned().unwrap_or(IrType::Unknown)
                        }
                        _ => IrType::Unknown,
                    };
                    let payload = builder.emit(
                        InstrKind::EnumPayload { value, index: 0 },
                        payload_ty,
                        Some(span),
                    );
                    self.lower_pattern_tests(builder, inner, payload, bindings, tests, span);
                }
            }
            Pattern::Tuple(elements) => {
                let element_ty = element_type(builder.value_type(value));
                for (index, element) in elements.iter().enumerate() {
                    let extracted = builder.emit(
                        InstrKind::Extract {
                            tensor: value,
                            index: index as u64,
                        },
                        element_ty.clone(),
                        Some(span),
                    );
                    self.lower_pattern_tests(builder, element, extracted, bindings, tests, span);
                }
            }
        }
    }

    /// Emits the boolean test for one deferred pattern test.
    fn lower_test(
        &mut self,
        builder: &mut FunctionBuilder,
        test: &MatchTest,
        span: Span,
    ) -> ValueId {
        match test {
            MatchTest::Literal { value, literal } => {
                let literal = self.lower_literal(builder, literal);
                builder.emit(
                    InstrKind::Binary {
                        op: BinaryOp::Eq,
                        lhs: *value,
                        rhs: literal,
                    },
                    IrType::Bool,
                    Some(span),
                )
            }
            MatchTest::Variant { value, variant } => builder.emit(
                InstrKind::EnumIs {
                    value: *value,
                    variant: variant.clone(),
                },
                IrType::Bool,
                Some(span),
            ),
        }
    }

    // ── helpers ─────────────────────────────────────────────────────

    fn env_value(&self, name: &str) -> ValueId {
        self.env
            .get(name)
            .copied()
            .unwrap_or_else(|| panic!("carried variable `{name}` missing from environment"))
    }

    fn env_type(&self, builder: &FunctionBuilder, name: &str) -> IrType {
        builder.value_type(self.env_value(name)).clone()
    }

    fn carried_variables(&self, body: &Expr) -> Vec<String> {
        let mut assigned = BTreeSet::new();
        collect_assigned_expr(body, &mut assigned);
        assigned
            .into_iter()
            .filter(|name| self.env.contains_key(name))
            .collect()
    }
}

fn collect_assigned_expr(expr: &Expr, out: &mut BTreeSet<String>) {
    match expr {
        Expr::Binary {
            op: BinaryOp::Assign,
            lhs,
            rhs,
            ..
        } => {
            if let Expr::Ident(ident) = lhs.as_ref() {
                out.insert(ident.name.clone());
            }
            collect_assigned_expr(lhs, out);
            collect_assigned_expr(rhs, out);
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_assigned_expr(lhs, out);
            collect_assigned_expr(rhs, out);
        }
        Expr::Unary { operand, .. } => collect_assigned_expr(operand, out),
        Expr::Array { elements, .. } => {
            for element in elements {
                collect_assigned_expr(element, out);
            }
        }
        Expr::Call { callee, args, .. } => {
            collect_assigned_expr(callee, out);
            for arg in args {
                collect_assigned_expr(arg, out);
            }
        }
        Expr::Index { target, index, .. } => {
            collect_assigned_expr(target, out);
            collect_assigned_expr(index, out);
        }
        Expr::Field { target, .. } => collect_assigned_expr(target, out),
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            collect_assigned_expr(condition, out);
            collect_assigned_expr(then_branch, out);
            if let Some(else_branch) = else_branch {
                collect_assigned_expr(else_branch, out);
            }
        }
        Expr::Block { items, .. } => {
            for item in items {
                collect_assigned_item(item, out);
            }
        }
        Expr::Match { expr, arms, .. } => {
            collect_assigned_expr(expr, out);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    collect_assigned_expr(guard, out);
                }
                collect_assigned_expr(&arm.body, out);
            }
        }
        Expr::Try { operand, .. } => collect_assigned_expr(operand, out),
        Expr::Literal { .. } | Expr::Ident(_) => {}
    }
}

fn collect_assigned_item(item: &Item, out: &mut BTreeSet<String>) {
    match item {
        Item::Expr(expr) => collect_assigned_expr(expr, out),
        Item::Let(decl) => collect_assigned_expr(&decl.value, out),
        Item::Tensor(decl) => collect_assigned_expr(&decl.value, out),
        Item::Return {
            value: Some(value), ..
        } => collect_assigned_expr(value, out),
        Item::Return { value: None, .. } => {}
        Item::While {
            condition, body, ..
        } => {
            collect_assigned_expr(condition, out);
            collect_assigned_expr(body, out);
        }
        Item::For { iterable, body, .. } => {
            collect_assigned_expr(iterable, out);
            collect_assigned_expr(body, out);
        }
        _ => {}
    }
}

fn field_value(
    lowerer: &mut Lowerer,
    builder: &mut FunctionBuilder,
    fields: &[Field],
    name: &str,
) -> Option<ValueId> {
    fields
        .iter()
        .find(|field| field.name.name == name)
        .map(|field| lowerer.lower_expr(builder, &field.value))
}

/// Result type of a call that resolves to a built-in rather than a user
/// function. Unknown callees fall through to `IrType::Unit`.
fn builtin_ret_type(name: &str) -> Option<IrType> {
    use apeiron_ast::builtins::{self, BuiltinRet};
    let signature = builtins::find(name)?;
    Some(match signature.ret {
        BuiltinRet::Unit => IrType::Unit,
        BuiltinRet::I64 => IrType::I64,
        BuiltinRet::F32 => IrType::F32,
        BuiltinRet::F64 => IrType::F64,
        BuiltinRet::Bool => IrType::Bool,
        BuiltinRet::Tensor => IrType::Tensor(TensorTy::new(ElemType::F32, vec![Dim::Dynamic])),
    })
}

fn ir_type(ty: &TypeExpr) -> IrType {
    if ty.dims.is_empty() {
        match ty.name.name.as_str() {
            "Option" | "Result" => {
                return IrType::Enum(EnumTy::new(&ty.name.name, Vec::new()));
            }
            _ => {}
        }
    }
    match ty.name.name.as_str() {
        "Option" => {
            let payload = ty.dims.iter().map(ir_type_from_dim).collect();
            IrType::Enum(EnumTy::new("Option", payload))
        }
        "Result" => {
            let payload = ty.dims.iter().map(ir_type_from_dim).collect();
            IrType::Enum(EnumTy::new("Result", payload))
        }
        _ if !ty.dims.is_empty() => {
            let elem = elem_type_for(&ty.name.name);
            let shape = ty
                .dims
                .iter()
                .map(|dim| match dim {
                    TypeDim::Const { value, .. } => Dim::Const(*value),
                    TypeDim::Dynamic(_) => Dim::Dynamic,
                })
                .collect();
            IrType::Tensor(TensorTy::new(elem, shape))
        }
        name => ir_type_from_name(name),
    }
}

/// Resolves a bare (unshaped) type name to an IR type.
fn ir_type_from_name(name: &str) -> IrType {
    match name {
        "bool" => IrType::Bool,
        "str" | "string" => IrType::Str,
        "unit" | "()" => IrType::Unit,
        _ => IrType::scalar(elem_type_for(name)),
    }
}

/// Resolves one generic argument such as the `i64` in `Option[i64]`.
fn ir_type_from_dim(dim: &TypeDim) -> IrType {
    match dim {
        TypeDim::Dynamic(ident) => ir_type_from_name(&ident.name),
        TypeDim::Const { .. } => IrType::F32,
    }
}

/// Maps a numeric type name to its element type; defaults to `f32`.
fn elem_type_for(name: &str) -> ElemType {
    match name {
        "i8" | "i16" | "i32" | "i64" | "int" => ElemType::I64,
        "f64" | "double" => ElemType::F64,
        _ => ElemType::F32,
    }
}

fn init_type(expr: &Expr) -> Option<IrType> {
    if let Some(literal) = tensor_literal_from_expr(expr) {
        let shape = literal
            .shape
            .iter()
            .map(|extent| Dim::Const(*extent))
            .collect();
        return Some(IrType::Tensor(TensorTy::new(literal.elem, shape)));
    }
    match expr {
        Expr::Literal {
            value: Literal::Int(_),
            ..
        } => Some(IrType::I64),
        Expr::Literal {
            value: Literal::Float(_),
            ..
        } => Some(IrType::F32),
        Expr::Literal {
            value: Literal::Bool(_),
            ..
        } => Some(IrType::Bool),
        Expr::Literal {
            value: Literal::Str(_),
            ..
        } => Some(IrType::Str),
        _ => None,
    }
}

fn init_of(expr: &Expr) -> Option<GlobalInit> {
    if let Some(literal) = tensor_literal_from_expr(expr) {
        return Some(GlobalInit::Tensor(literal));
    }
    match expr {
        Expr::Literal {
            value: Literal::Int(value),
            ..
        } => Some(GlobalInit::Const(ConstValue::Int(*value))),
        Expr::Literal {
            value: Literal::Float(value),
            ..
        } => Some(GlobalInit::Const(ConstValue::Float(*value))),
        Expr::Literal {
            value: Literal::Bool(value),
            ..
        } => Some(GlobalInit::Const(ConstValue::Bool(*value))),
        Expr::Literal {
            value: Literal::Str(value),
            ..
        } => Some(GlobalInit::Const(ConstValue::Str(value.clone()))),
        _ => None,
    }
}

fn tensor_literal(elements: &[Expr]) -> Option<TensorLit> {
    let mut elem = ElemType::F32;
    let mut shape: Option<Vec<u64>> = None;
    let mut data = Vec::new();
    for element in elements {
        let literal = tensor_literal_from_expr(element)?;
        match &shape {
            None => shape = Some(literal.shape.clone()),
            Some(expected) if *expected != literal.shape => return None,
            Some(_) => {}
        }
        elem = unify_elem(elem, literal.elem);
        data.extend(literal.data);
    }
    let mut full = vec![elements.len() as u64];
    full.extend(shape?);
    Some(TensorLit {
        elem,
        shape: full,
        data,
    })
}

fn tensor_literal_from_expr(expr: &Expr) -> Option<TensorLit> {
    match expr {
        Expr::Array { elements, .. } => tensor_literal(elements),
        Expr::Literal {
            value: Literal::Int(value),
            ..
        } => Some(TensorLit {
            elem: ElemType::I64,
            shape: Vec::new(),
            data: vec![*value as f64],
        }),
        Expr::Literal {
            value: Literal::Float(value),
            ..
        } => Some(TensorLit {
            elem: ElemType::F32,
            shape: Vec::new(),
            data: vec![*value],
        }),
        _ => None,
    }
}

fn unify_elem(a: ElemType, b: ElemType) -> ElemType {
    use ElemType::*;
    match (a, b) {
        (F64, _) | (_, F64) => F64,
        (F32, _) | (_, F32) => F32,
        (I64, I64) => I64,
    }
}

/// Placeholder value returned when a diverging block (`return`/`break`)
/// terminates before producing a result. Callers on a terminated block
/// ignore it; the verifier catches any accidental use.
fn placeholder_value(builder: &FunctionBuilder) -> ValueId {
    builder.params().first().copied().unwrap_or(ValueId(0))
}

fn element_type(ty: &IrType) -> IrType {
    match ty {
        IrType::Tensor(tensor) if tensor.rank() <= 1 => IrType::scalar(tensor.elem),
        IrType::Tensor(tensor) => {
            IrType::Tensor(TensorTy::new(tensor.elem, tensor.shape[1..].to_vec()))
        }
        other => other.clone(),
    }
}

fn make_tensor_type(builder: &FunctionBuilder, elements: &[ValueId]) -> IrType {
    let mut elem = ElemType::F32;
    let mut inner: Option<Vec<Dim>> = None;
    for &element in elements {
        let ty = builder.value_type(element);
        let (element_elem, shape) = match ty {
            IrType::Tensor(tensor) => (tensor.elem, tensor.shape.clone()),
            IrType::I64 => (ElemType::I64, Vec::new()),
            IrType::F32 => (ElemType::F32, Vec::new()),
            IrType::F64 => (ElemType::F64, Vec::new()),
            _ => (ElemType::F32, Vec::new()),
        };
        elem = unify_elem(elem, element_elem);
        if inner.is_none() {
            inner = Some(shape);
        }
    }
    let mut shape = vec![Dim::Const(elements.len() as u64)];
    shape.extend(inner.unwrap_or_default());
    IrType::Tensor(TensorTy::new(elem, shape))
}

fn matmul_type(a: &IrType, b: &IrType) -> IrType {
    let (lhs, rhs) = match (a, b) {
        (IrType::Tensor(lhs), IrType::Tensor(rhs)) => (lhs, rhs),
        _ => return IrType::F32,
    };
    let elem = unify_elem(lhs.elem, rhs.elem);
    let shape = match (lhs.rank(), rhs.rank()) {
        (2, 2) => vec![lhs.shape[0], rhs.shape[1]],
        (2, 1) => vec![lhs.shape[0]],
        (1, 2) => vec![rhs.shape[1]],
        (1, 1) => Vec::new(),
        _ => return IrType::Tensor(TensorTy::new(elem, Vec::new())),
    };
    if shape.is_empty() {
        IrType::scalar(elem)
    } else {
        IrType::Tensor(TensorTy::new(elem, shape))
    }
}

fn common_type(a: &IrType, b: &IrType) -> IrType {
    if a == b {
        return a.clone();
    }
    if matches!(a, IrType::Unknown) {
        return b.clone();
    }
    if matches!(b, IrType::Unknown) {
        return a.clone();
    }
    match (a, b) {
        (IrType::Tensor(lhs), IrType::Tensor(rhs)) => {
            let shape = if lhs.rank() >= rhs.rank() {
                lhs.shape.clone()
            } else {
                rhs.shape.clone()
            };
            IrType::Tensor(TensorTy::new(unify_elem(lhs.elem, rhs.elem), shape))
        }
        (IrType::Enum(lhs), IrType::Enum(rhs)) if lhs.name == rhs.name => {
            let len = lhs.payload.len().max(rhs.payload.len());
            let payload = (0..len)
                .map(
                    |index| match (lhs.payload.get(index), rhs.payload.get(index)) {
                        (Some(a), Some(b)) => common_type(a, b),
                        (Some(a), None) => a.clone(),
                        (None, Some(b)) => b.clone(),
                        (None, None) => IrType::F32,
                    },
                )
                .collect();
            IrType::Enum(EnumTy::new(&lhs.name, payload))
        }
        _ => {
            if a.is_numeric() {
                a.clone()
            } else if b.is_numeric() {
                b.clone()
            } else {
                a.clone()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apeiron_parser::parse_source;

    fn lower(source: &str) -> Module {
        let program = parse_source(source, apeiron_core::source::FileId(0)).expect("parses");
        lower_program(&program).expect("lowers")
    }

    #[test]
    fn lowers_tensor_pipeline() {
        let module = lower(
            r#"
tensor W : f32[2, 3] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
tensor x : f32[3] = [1.0, 2.0, 3.0]
tensor y = W @ x
"#,
        );
        let main = module.function("main").expect("main exists");
        assert!(main.block_count() >= 1);
        assert!(module.instruction_count() > 0);
    }

    #[test]
    fn lowers_function_call() {
        let module = lower(
            r#"
fn scale(x: f32) -> f32 { x * 2.0 }
let y = scale(3.0)
"#,
        );
        assert!(module.function("scale").is_some());
        assert!(module.function("main").is_some());
    }
}

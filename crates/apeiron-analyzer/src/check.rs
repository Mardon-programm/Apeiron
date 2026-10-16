//! Declaration checking and validation of the cognitive primitives.
//!
//! Every item is checked exactly once by [`Analyzer::check_item`], which
//! also returns the type the item evaluates to so that block expressions
//! can pick up their final value. Field-based declarations share a small
//! specification table (`required`/optional entries and their expected
//! value shape) so that `belief`, `plan`, `route`, and friends validate
//! consistently.

use apeiron_ast::{
    ArchOp, ArchitectureBlock, BeliefStmt, FnDecl, Ident, Item, LetDecl, MemoryDecl, SocietyBlock,
    SocietyMember, TensorDecl,
};
use apeiron_core::span::Span;

use crate::symbol::{ScopeKind, Symbol, SymbolKind};
use crate::types::{DomainKind, TensorType, Type};
use crate::Analyzer;

use super::infer::{is_number, is_text};

/// Expected value of a `name: value` entry.
#[derive(Clone, Copy)]
enum FieldExpect {
    Any,
    Number,
    Text,
}

/// One row of a field specification table.
struct FieldSpec {
    name: &'static str,
    required: bool,
    expect: FieldExpect,
}

const fn required(name: &'static str, expect: FieldExpect) -> FieldSpec {
    FieldSpec {
        name,
        required: true,
        expect,
    }
}

const fn optional(name: &'static str, expect: FieldExpect) -> FieldSpec {
    FieldSpec {
        name,
        required: false,
        expect,
    }
}

const OPTIMIZER_KINDS: &[&str] = &["adam", "adamw", "sgd", "rmsprop", "adagrad", "adadelta"];

impl Analyzer {
    /// Checks an item, returning the type it evaluates to.
    pub(crate) fn check_item(&mut self, item: &Item) -> Type {
        match item {
            Item::Fn(decl) => self.check_fn(decl),
            Item::Let(decl) => self.check_let(decl),
            Item::Tensor(decl) => self.check_tensor(decl),
            Item::Return { value, span } => {
                self.check_return(value.as_ref(), *span);
                Type::Unit
            }
            Item::While {
                condition, body, ..
            } => {
                self.check_while(condition, body);
                Type::Unit
            }
            Item::For {
                variable,
                iterable,
                body,
                ..
            } => {
                self.check_for(variable, iterable, body);
                Type::Unit
            }
            Item::Break { span } => {
                self.check_loop_control("break", *span);
                Type::Unit
            }
            Item::Continue { span } => {
                self.check_loop_control("continue", *span);
                Type::Unit
            }
            Item::Expr(expr) => self.infer_expr(expr),
            Item::Optimizer {
                kind,
                settings,
                span,
            } => {
                self.check_optimizer(kind, settings, *span);
                Type::Unit
            }
            Item::Train { model, dataset, .. } => {
                self.check_train(model, dataset);
                Type::Unit
            }
            Item::Gradient {
                target,
                with_respect_to,
                ..
            } => {
                self.check_gradient(target, with_respect_to);
                Type::Unit
            }
            Item::Distribution {
                name,
                categories,
                span,
            } => self.check_distribution(name, categories, *span),
            Item::Belief(stmt) => self.check_belief(stmt),
            Item::Cause { from, to, .. } => {
                self.register_causal_node(&from.name, from.span);
                self.register_causal_node(&to.name, to.span);
                Type::Unit
            }
            Item::Observe { target, value, .. } => {
                self.check_observation("observe", target, value);
                Type::Unit
            }
            Item::Intervene { target, value, .. } => {
                self.check_observation("intervene", target, value);
                Type::Unit
            }
            Item::Counterfactual {
                target, condition, ..
            } => {
                self.check_counterfactual(target, condition);
                Type::Unit
            }
            Item::Simulate {
                parameter, value, ..
            } => {
                let value_ty = self.infer_expr(value);
                self.expect_field(
                    &value_ty,
                    FieldExpect::Number,
                    value,
                    "simulate",
                    &parameter.name,
                );
                Type::Unit
            }
            Item::Memory(decl) => self.check_memory(decl),
            Item::Remember {
                memory, entries, ..
            } => {
                self.require_memory(memory, "remember");
                for field in entries {
                    let _ = self.infer_expr(&field.value);
                }
                Type::Unit
            }
            Item::Consolidate {
                from,
                to,
                threshold,
                ..
            } => {
                self.require_memory(from, "consolidate");
                self.require_memory(to, "consolidate");
                if let Some(threshold) = threshold {
                    let threshold_ty = self.infer_expr(threshold);
                    self.expect_field(
                        &threshold_ty,
                        FieldExpect::Number,
                        threshold,
                        "consolidate",
                        "threshold",
                    );
                }
                Type::Unit
            }
            Item::Recall {
                memory,
                arguments,
                span,
            } => {
                self.require_memory(memory, "recall");
                self.check_open_fields(arguments, FieldExpect::Any, "recall arguments", *span);
                Type::Unknown
            }
            Item::Plan { name, body, span } => self.check_plan(name, body, *span),
            Item::SelfModel { name, body, span } => self.check_self_model(name, body, *span),
            Item::Architecture(block) => self.check_architecture(block),
            Item::Society(block) => self.check_society(block),
            Item::Route {
                request,
                cluster,
                budget,
                span,
            } => {
                self.check_route(request, cluster, budget, *span);
                Type::Unit
            }
        }
    }

    // ── bindings & functions ────────────────────────────────────────

    fn check_fn(&mut self, decl: &FnDecl) -> Type {
        let param_types: Vec<Type> = decl
            .params
            .iter()
            .map(|param| match &param.ty {
                Some(ty) => self.type_from_typeexpr(ty),
                None => Type::Unknown,
            })
            .collect();
        let return_type = match &decl.return_type {
            Some(ty) => self.type_from_typeexpr(ty),
            None => Type::Unknown,
        };

        let fn_type = Type::Fn {
            params: param_types.clone(),
            ret: Box::new(return_type.clone()),
        };
        self.declare(Symbol::new(
            decl.name.name.as_str(),
            SymbolKind::Fn,
            fn_type,
            decl.name.span,
        ));

        self.push_scope(ScopeKind::Function);
        for (param, ty) in decl.params.iter().zip(param_types) {
            self.declare(Symbol::new(
                param.name.name.as_str(),
                SymbolKind::Param,
                ty,
                param.name.span,
            ));
        }

        let previous_return = self.function_return.replace(return_type.clone());
        let previous_loop = std::mem::replace(&mut self.loop_depth, 0);
        let body_ty = self.infer_expr(&decl.body);
        self.loop_depth = previous_loop;
        self.function_return = previous_return;

        if !return_type.is_sentinel() && !return_type.accepts(&body_ty) {
            self.error(
                "AN0003",
                format!(
                    "function `{}` returns {body_ty} but declares {return_type}",
                    decl.name.name
                ),
                decl.body.span(),
            );
        }
        self.pop_scope();
        Type::Unit
    }

    fn check_let(&mut self, decl: &LetDecl) -> Type {
        let value_ty = self.infer_expr(&decl.value);
        let bound = match &decl.ty {
            Some(annot) => {
                let declared = self.type_from_typeexpr(annot);
                if !declared.accepts(&value_ty) {
                    self.error(
                        "AN0003",
                        format!(
                            "cannot bind {value_ty} to `{}` of type {declared}",
                            decl.name.name
                        ),
                        decl.value.span(),
                    );
                }
                declared
            }
            None => value_ty,
        };
        self.declare(Symbol::new(
            decl.name.name.as_str(),
            SymbolKind::Variable,
            bound,
            decl.name.span,
        ));
        Type::Unit
    }

    fn check_tensor(&mut self, decl: &TensorDecl) -> Type {
        let inferred = self.infer_expr(&decl.value);
        let bound = match &decl.ty {
            Some(annot) => {
                let declared = self.type_from_typeexpr(annot);
                if !declared.accepts(&inferred) {
                    self.error(
                        "AN0004",
                        format!(
                            "initializer has type {inferred} but `{}` is declared {declared}",
                            decl.name.name
                        ),
                        decl.value.span(),
                    );
                }
                declared
            }
            None => inferred,
        };
        if matches!(bound, Type::Bool | Type::Str | Type::Unit) {
            self.error(
                "AN0006",
                format!(
                    "`{}` must be a numeric tensor, found {bound}",
                    decl.name.name
                ),
                decl.span,
            );
        }
        self.declare(Symbol::new(
            decl.name.name.as_str(),
            SymbolKind::for_tensor(decl.kind),
            bound,
            decl.name.span,
        ));
        Type::Unit
    }

    fn check_return(&mut self, value: Option<&apeiron_ast::Expr>, span: Span) {
        let Some(expected) = self.function_return.clone() else {
            self.error("AN0010", "`return` used outside of a function", span);
            if let Some(value) = value {
                let _ = self.infer_expr(value);
            }
            return;
        };
        match value {
            Some(value) => {
                let actual = self.infer_expr(value);
                if !expected.is_sentinel() && !expected.accepts(&actual) {
                    self.error(
                        "AN0003",
                        format!("returning {actual} from a function declared to return {expected}"),
                        value.span(),
                    );
                }
            }
            None => {
                if !expected.is_sentinel() && expected != Type::Unit {
                    self.error(
                        "AN0003",
                        format!("returning `()` from a function declared to return {expected}"),
                        span,
                    );
                }
            }
        }
    }

    // ── control flow ────────────────────────────────────────────────

    fn check_while(&mut self, condition: &apeiron_ast::Expr, body: &apeiron_ast::Expr) {
        let condition_ty = self.infer_expr(condition);
        if !condition_ty.is_sentinel() && condition_ty != Type::Bool {
            self.error(
                "AN0003",
                format!("`while` condition must be bool, found {condition_ty}"),
                condition.span(),
            );
        }
        self.loop_depth += 1;
        let _ = self.infer_expr(body);
        self.loop_depth -= 1;
    }

    fn check_for(
        &mut self,
        variable: &Ident,
        iterable: &apeiron_ast::Expr,
        body: &apeiron_ast::Expr,
    ) {
        let iterable_ty = self.infer_expr(iterable);
        let element = self.iter_element(&iterable_ty, iterable.span());
        self.push_scope(ScopeKind::Block);
        self.declare(Symbol::new(
            variable.name.as_str(),
            SymbolKind::Variable,
            element,
            variable.span,
        ));
        self.loop_depth += 1;
        let _ = self.infer_expr(body);
        self.loop_depth -= 1;
        self.pop_scope();
    }

    fn iter_element(&mut self, ty: &Type, span: Span) -> Type {
        match ty {
            Type::Tensor(tensor) if tensor.rank() >= 1 => {
                if tensor.rank() == 1 {
                    Type::scalar(tensor.elem)
                } else {
                    Type::Tensor(TensorType::new(tensor.elem, tensor.shape[1..].to_vec()))
                }
            }
            Type::Error | Type::Unknown => ty.clone(),
            other => {
                self.error("AN0004", format!("cannot iterate over {other}"), span);
                Type::Error
            }
        }
    }

    fn check_loop_control(&mut self, keyword: &str, span: Span) {
        if self.loop_depth == 0 {
            self.error(
                "AN0016",
                format!("`{keyword}` used outside of a loop"),
                span,
            );
        }
    }

    // ── training & data ─────────────────────────────────────────────

    fn check_optimizer(&mut self, kind: &Ident, settings: &[apeiron_ast::Field], span: Span) {
        if !OPTIMIZER_KINDS.contains(&kind.name.as_str()) {
            self.error(
                "AN0006",
                format!("unknown optimizer `{}`", kind.name),
                kind.span,
            );
        }
        let specs = [
            optional("lr", FieldExpect::Number),
            optional("momentum", FieldExpect::Number),
            optional("weight_decay", FieldExpect::Number),
            optional("beta1", FieldExpect::Number),
            optional("beta2", FieldExpect::Number),
            optional("eps", FieldExpect::Number),
        ];
        self.check_fields(settings, &specs, "optimizer", span);
    }

    fn check_train(&mut self, model: &Ident, dataset: &Ident) {
        match self.lookup(&model.name) {
            Some(symbol) if matches!(symbol.kind, SymbolKind::Tensor | SymbolKind::Parameter) => {}
            Some(symbol) => self.error(
                "AN0003",
                format!(
                    "`train` expects a tensor or parameter, found {} `{}`",
                    symbol.kind, model.name
                ),
                model.span,
            ),
            None => self.unresolved(&model.name, model.span),
        }
        if self.lookup(&dataset.name).is_none() {
            self.unresolved(&dataset.name, dataset.span);
        }
    }

    fn check_gradient(&mut self, target: &Ident, with_respect_to: &Ident) {
        match self.lookup(&target.name) {
            Some(symbol) if symbol.ty.numeric_elem().is_some() || symbol.ty.is_sentinel() => {}
            Some(symbol) => self.error(
                "AN0003",
                format!(
                    "`gradient` target `{}` must be numeric, found {}",
                    target.name, symbol.ty
                ),
                target.span,
            ),
            None => self.unresolved(&target.name, target.span),
        }
        match self.lookup(&with_respect_to.name) {
            Some(symbol)
                if symbol.ty.is_tensor()
                    || matches!(symbol.kind, SymbolKind::Tensor | SymbolKind::Parameter)
                    || symbol.ty.is_sentinel() => {}
            Some(symbol) => self.error(
                "AN0003",
                format!(
                    "`gradient` must differentiate a tensor, found `{}` of type {}",
                    with_respect_to.name, symbol.ty
                ),
                with_respect_to.span,
            ),
            None => self.unresolved(&with_respect_to.name, with_respect_to.span),
        }
    }

    // ── epistemology ────────────────────────────────────────────────

    fn check_distribution(
        &mut self,
        name: &Ident,
        categories: &[apeiron_ast::Field],
        span: Span,
    ) -> Type {
        if categories.is_empty() {
            self.error(
                "AN0008",
                format!(
                    "distribution `{}` must declare at least one category",
                    name.name
                ),
                span,
            );
        }
        for field in categories {
            let ty = self.infer_expr(&field.value);
            self.expect_field(
                &ty,
                FieldExpect::Number,
                &field.value,
                "distribution",
                &field.name.name,
            );
        }
        self.declare(Symbol::new(
            name.name.as_str(),
            SymbolKind::Distribution,
            Type::Domain(DomainKind::Distribution),
            name.span,
        ));
        Type::Domain(DomainKind::Distribution)
    }

    fn check_belief(&mut self, stmt: &BeliefStmt) -> Type {
        let name = stmt.name();
        let fields = stmt.fields();
        let span = stmt.span();
        match stmt {
            BeliefStmt::Declare { .. } => {
                let specs = [
                    required("proposition", FieldExpect::Any),
                    required("confidence", FieldExpect::Number),
                    optional("because", FieldExpect::Any),
                    optional("sources", FieldExpect::Any),
                    optional("prior", FieldExpect::Number),
                ];
                self.check_fields(fields, &specs, "belief declaration", span);
                self.declare(Symbol::new(
                    name.name.as_str(),
                    SymbolKind::Belief,
                    Type::Domain(DomainKind::Belief),
                    name.span,
                ));
            }
            BeliefStmt::Update { .. } => {
                let specs = [
                    required("likelihood", FieldExpect::Number),
                    required("reliability", FieldExpect::Number),
                    optional("evidence", FieldExpect::Any),
                ];
                self.check_fields(fields, &specs, "belief update", span);
                match self.lookup(&name.name) {
                    Some(symbol) if symbol.kind == SymbolKind::Belief => {}
                    Some(symbol) => self.error(
                        "AN0003",
                        format!("`{}` is a {}, not a belief", name.name, symbol.kind),
                        name.span,
                    ),
                    None => self.error(
                        "AN0003",
                        format!("cannot update unknown belief `{}`", name.name),
                        name.span,
                    ),
                }
            }
        }
        Type::Domain(DomainKind::Belief)
    }

    // ── causality ───────────────────────────────────────────────────

    fn register_causal_node(&mut self, name: &str, span: Span) {
        if self.causal_nodes.insert(name.to_string()) && self.scopes[0].get(name).is_none() {
            self.scopes[0].insert(Symbol::new(
                name,
                SymbolKind::CausalNode,
                Type::Domain(DomainKind::CausalNode),
                span,
            ));
        }
    }

    fn check_observation(&mut self, keyword: &str, target: &Ident, value: &apeiron_ast::Expr) {
        let value_ty = self.infer_expr(value);
        self.expect_field(&value_ty, FieldExpect::Number, value, keyword, &target.name);
        if !self.causal_nodes.contains(&target.name) && self.lookup(&target.name).is_none() {
            self.warning(
                "AN0012",
                format!(
                    "`{keyword}` references unknown causal node `{}`",
                    target.name
                ),
                target.span,
            );
        }
    }

    fn check_counterfactual(&mut self, target: &Ident, condition: &apeiron_ast::Expr) {
        let condition_ty = self.infer_expr(condition);
        if !condition_ty.is_sentinel() && condition_ty != Type::Bool {
            self.error(
                "AN0003",
                format!("`counterfactual` condition must be bool, found {condition_ty}"),
                condition.span(),
            );
        }
        if !self.causal_nodes.contains(&target.name) && self.lookup(&target.name).is_none() {
            self.warning(
                "AN0012",
                format!(
                    "`counterfactual` references unknown causal node `{}`",
                    target.name
                ),
                target.span,
            );
        }
    }

    // ── memory & cognition ──────────────────────────────────────────

    fn check_memory(&mut self, decl: &MemoryDecl) -> Type {
        let types: Vec<Type> = decl
            .fields
            .iter()
            .map(|field| self.infer_expr(&field.value))
            .collect();
        match decl
            .fields
            .iter()
            .position(|field| field.name.name == "dimension")
        {
            Some(index) => self.expect_field(
                &types[index],
                FieldExpect::Number,
                &decl.fields[index].value,
                "memory",
                "dimension",
            ),
            None => self.error(
                "AN0008",
                "memory store is missing required field `dimension`",
                decl.span,
            ),
        }
        self.declare(Symbol::new(
            decl.name.name.as_str(),
            SymbolKind::Memory(decl.tier),
            Type::Domain(DomainKind::MemoryStore),
            decl.name.span,
        ));
        Type::Domain(DomainKind::MemoryStore)
    }

    fn require_memory(&mut self, memory: &Ident, keyword: &str) {
        match self.lookup(&memory.name) {
            Some(symbol) if matches!(symbol.kind, SymbolKind::Memory(_)) => {}
            Some(symbol) => self.error(
                "AN0011",
                format!(
                    "`{keyword}` expects a memory store, found {} `{}`",
                    symbol.kind, memory.name
                ),
                memory.span,
            ),
            None => self.warning(
                "AN0011",
                format!(
                    "`{keyword}` references unknown memory store `{}`",
                    memory.name
                ),
                memory.span,
            ),
        }
    }

    fn check_plan(&mut self, name: &Ident, body: &[apeiron_ast::Field], span: Span) -> Type {
        let specs = [
            required("goal", FieldExpect::Any),
            optional("horizon", FieldExpect::Number),
            optional("top_k", FieldExpect::Number),
            optional("priority", FieldExpect::Any),
        ];
        self.check_fields(body, &specs, "plan", span);
        self.declare(Symbol::new(
            name.name.as_str(),
            SymbolKind::Plan,
            Type::Domain(DomainKind::Plan),
            name.span,
        ));
        Type::Domain(DomainKind::Plan)
    }

    fn check_self_model(&mut self, name: &Ident, body: &[apeiron_ast::Field], span: Span) -> Type {
        self.check_open_fields(body, FieldExpect::Any, "self_model", span);
        self.declare(Symbol::new(
            name.name.as_str(),
            SymbolKind::SelfModel,
            Type::Domain(DomainKind::SelfModel),
            name.span,
        ));
        Type::Domain(DomainKind::SelfModel)
    }

    // ── architecture & societies ────────────────────────────────────

    fn check_architecture(&mut self, block: &ArchitectureBlock) -> Type {
        self.declare(Symbol::new(
            block.name.name.as_str(),
            SymbolKind::Architecture,
            Type::Domain(DomainKind::Architecture),
            block.name.span,
        ));
        for operation in &block.operations {
            match operation {
                ArchOp::AddComponent { component, .. } => {
                    let ty = self.infer_expr(component);
                    if !is_text(&ty) {
                        self.error(
                            "AN0003",
                            format!("`add component` expects text, found {ty}"),
                            component.span(),
                        );
                    }
                }
                ArchOp::ExpandMemory { memory, by, .. } => {
                    match memory.as_str_literal() {
                        Some(name) if self.lookup(name).is_none() => self.warning(
                            "AN0013",
                            format!("`expand memory` references unknown memory store \"{name}\""),
                            memory.span(),
                        ),
                        Some(_) => {}
                        None => {
                            let ty = self.infer_expr(memory);
                            if !is_text(&ty) {
                                self.error(
                                    "AN0003",
                                    format!("`expand memory` expects a memory name, found {ty}"),
                                    memory.span(),
                                );
                            }
                        }
                    }
                    let by_ty = self.infer_expr(by);
                    self.expect_field(&by_ty, FieldExpect::Number, by, "expand memory", "by");
                }
                ArchOp::ChangeRouting { routing, .. } => {
                    let ty = self.infer_expr(routing);
                    if !is_text(&ty) {
                        self.error(
                            "AN0003",
                            format!("`change routing` expects text, found {ty}"),
                            routing.span(),
                        );
                    }
                }
            }
        }
        Type::Domain(DomainKind::Architecture)
    }

    fn check_society(&mut self, block: &SocietyBlock) -> Type {
        self.declare(Symbol::new(
            block.name.name.as_str(),
            SymbolKind::Society,
            Type::Domain(DomainKind::Society),
            block.name.span,
        ));

        self.push_scope(ScopeKind::Society);
        for member in &block.members {
            match member {
                SocietyMember::Agent { name, fields, span } => {
                    let specs = [required("role", FieldExpect::Text)];
                    self.check_fields(fields, &specs, "agent", *span);
                    self.declare(Symbol::new(
                        name.name.as_str(),
                        SymbolKind::Agent,
                        Type::Domain(DomainKind::Agent),
                        name.span,
                    ));
                }
                SocietyMember::Cluster { name, .. } => {
                    self.clusters.insert(name.name.clone());
                    self.declare(Symbol::new(
                        name.name.as_str(),
                        SymbolKind::Cluster,
                        Type::Domain(DomainKind::Cluster),
                        name.span,
                    ));
                }
            }
        }

        for member in &block.members {
            if let SocietyMember::Cluster { name, members, .. } = member {
                for target in members {
                    match self.lookup(&target.name) {
                        Some(symbol) if symbol.kind == SymbolKind::Agent => {}
                        Some(symbol) => self.error(
                            "AN0014",
                            format!(
                                "cluster `{}` member `{}` is a {}, not an agent",
                                name.name, target.name, symbol.kind
                            ),
                            target.span,
                        ),
                        None => self.error(
                            "AN0014",
                            format!(
                                "cluster `{}` references unknown agent `{}`",
                                name.name, target.name
                            ),
                            target.span,
                        ),
                    }
                }
            }
        }
        self.pop_scope();
        Type::Domain(DomainKind::Society)
    }

    fn check_route(
        &mut self,
        request: &Ident,
        cluster: &Ident,
        budget: &[apeiron_ast::Field],
        span: Span,
    ) {
        let specs = [
            required("tokens", FieldExpect::Number),
            required("latency_ms", FieldExpect::Number),
            optional("cost", FieldExpect::Number),
            optional("max_retries", FieldExpect::Number),
        ];
        self.check_fields(budget, &specs, "route budget", span);
        if self.lookup(&request.name).is_none() {
            self.unresolved(&request.name, request.span);
        }
        match self.lookup(&cluster.name) {
            Some(symbol) if symbol.kind == SymbolKind::Cluster => {}
            Some(symbol) => self.error(
                "AN0015",
                format!(
                    "route target `{}` is a {}, not a cluster",
                    cluster.name, symbol.kind
                ),
                cluster.span,
            ),
            None => {
                if self.clusters.contains(&cluster.name) {
                    // Declared in a society whose scope has already closed.
                } else {
                    self.warning(
                        "AN0015",
                        format!("route target `{}` is not a known cluster", cluster.name),
                        cluster.span,
                    );
                }
            }
        }
    }

    // ── field tables ────────────────────────────────────────────────

    fn check_fields(
        &mut self,
        fields: &[apeiron_ast::Field],
        specs: &[FieldSpec],
        context: &str,
        span: Span,
    ) {
        let types: Vec<Type> = fields
            .iter()
            .map(|field| self.infer_expr(&field.value))
            .collect();
        for spec in specs {
            match fields.iter().position(|field| field.name.name == spec.name) {
                Some(index) => self.expect_field(
                    &types[index],
                    spec.expect,
                    &fields[index].value,
                    context,
                    spec.name,
                ),
                None if spec.required => self.error(
                    "AN0008",
                    format!("{context} is missing required field `{}`", spec.name),
                    span,
                ),
                None => {}
            }
        }
        for (index, field) in fields.iter().enumerate() {
            let _ = &types[index];
            if !specs.iter().any(|spec| spec.name == field.name.name) {
                self.warning(
                    "AN0008",
                    format!("unknown field `{}` in {context}", field.name.name),
                    field.name.span,
                );
            }
        }
    }

    fn check_open_fields(
        &mut self,
        fields: &[apeiron_ast::Field],
        expect: FieldExpect,
        context: &str,
        span: Span,
    ) {
        if fields.is_empty() {
            self.warning("AN0008", format!("{context} has no entries"), span);
        }
        for field in fields {
            let ty = self.infer_expr(&field.value);
            self.expect_field(&ty, expect, &field.value, context, &field.name.name);
        }
    }

    fn expect_field(
        &mut self,
        ty: &Type,
        expect: FieldExpect,
        value: &apeiron_ast::Expr,
        context: &str,
        name: &str,
    ) {
        match expect {
            FieldExpect::Any => {}
            FieldExpect::Number if !is_number(ty) => self.error(
                "AN0003",
                format!("field `{name}` of {context} must be numeric, found {ty}"),
                value.span(),
            ),
            FieldExpect::Number => {}
            FieldExpect::Text if !is_text(ty) => self.error(
                "AN0003",
                format!("field `{name}` of {context} must be text, found {ty}"),
                value.span(),
            ),
            FieldExpect::Text => {}
        }
    }
}

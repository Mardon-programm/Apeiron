//! Expression type inference and tensor shape inference.

use apeiron_ast::builtins::BuiltinRet;
use apeiron_ast::{BinaryOp, Expr, Ident, Literal, Pattern, TypeDim, TypeExpr, UnaryOp};
use apeiron_core::span::Span;
use apeiron_core::AbiType;

use crate::symbol::SymbolKind;
use crate::types::{ElemType, TensorType, Type};
use crate::Analyzer;

impl Analyzer {
    /// Infers the type of an expression, reporting diagnostics on failure.
    pub(crate) fn infer_expr(&mut self, expr: &Expr) -> Type {
        match expr {
            Expr::Literal { value, .. } => literal_type(value),
            Expr::Ident(ident) => {
                let found = self
                    .lookup(&ident.name)
                    .map(|symbol| (symbol.kind, symbol.ty.clone()));
                match found {
                    // A causal node names a continuous real-valued variable;
                    // reading it yields its value, not the opaque domain type.
                    Some((SymbolKind::CausalNode, _)) => Type::F32,
                    Some((_, ty)) => ty,
                    None => {
                        if ident.name == "None" {
                            Type::Option(Box::new(Type::Unknown))
                        } else {
                            self.unresolved(&ident.name, ident.span);
                            Type::Error
                        }
                    }
                }
            }
            Expr::Array { elements, span } => self.infer_array(elements, *span),
            Expr::Unary { op, operand, span } => {
                let operand_ty = self.infer_expr(operand);
                self.infer_unary(*op, operand_ty, *span)
            }
            Expr::Binary { op, lhs, rhs, span } => {
                let lhs_ty = self.infer_expr(lhs);
                let rhs_ty = self.infer_expr(rhs);
                self.infer_binary(*op, &lhs_ty, &rhs_ty, lhs, *span)
            }
            Expr::Call { callee, args, span } => self.infer_call(callee, args, *span),
            Expr::Index {
                target,
                index,
                span,
            } => self.infer_index(target, index, *span),
            Expr::Field { target, .. } => {
                // Field access is opaque for now; it never produces errors
                // so that `society.Agent` style references stay usable.
                let _ = self.infer_expr(target);
                Type::Unknown
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                span,
            } => self.infer_if(condition, then_branch, else_branch.as_deref(), *span),
            Expr::Block { items, .. } => self.infer_block(items),
            Expr::Match { expr, arms, span } => self.infer_match(expr, arms, *span),
            Expr::Try { operand, span } => self.infer_try(operand, *span),
        }
    }

    // ── literals & aggregates ───────────────────────────────────────

    /// Infers the element-wise shape of a (possibly nested) tensor literal.
    pub(crate) fn infer_array(&mut self, elements: &[Expr], span: Span) -> Type {
        if elements.is_empty() {
            self.error_with_help(
                "AN0005",
                "cannot infer the element type of an empty tensor literal",
                span,
                "annotate the tensor with an explicit type instead",
            );
            return Type::Error;
        }

        let mut elem: Option<ElemType> = None;
        let mut inner_shape: Option<Vec<Option<u64>>> = None;
        let mut failed = false;

        for element in elements {
            let ty = self.infer_expr(element);
            let (element_elem, element_shape) = match ty {
                Type::Tensor(tensor) => (tensor.elem, tensor.shape),
                Type::I64 => (ElemType::I64, Vec::new()),
                Type::F32 => (ElemType::F32, Vec::new()),
                Type::F64 => (ElemType::F64, Vec::new()),
                Type::Error | Type::Unknown => {
                    failed = true;
                    continue;
                }
                other => {
                    self.error(
                        "AN0005",
                        format!("tensor literal element must be numeric, found {other}"),
                        element.span(),
                    );
                    failed = true;
                    continue;
                }
            };

            match &inner_shape {
                None => inner_shape = Some(element_shape),
                Some(expected) if *expected != element_shape => {
                    self.error(
                        "AN0005",
                        format!(
                            "tensor literal is ragged: expected inner shape {}, found {}",
                            format_shape(expected),
                            format_shape(&element_shape)
                        ),
                        element.span(),
                    );
                    failed = true;
                }
                Some(_) => {}
            }

            elem = Some(match elem {
                None => element_elem,
                Some(previous) => match previous.unify(element_elem) {
                    Some(unified) => unified,
                    None => {
                        self.error(
                            "AN0005",
                            format!(
                                "tensor literal mixes incompatible element types {previous} and {element_elem}"
                            ),
                            element.span(),
                        );
                        failed = true;
                        element_elem
                    }
                },
            });
        }

        if failed {
            return Type::Error;
        }
        let mut shape = vec![Some(elements.len() as u64)];
        shape.extend(inner_shape.unwrap_or_default());
        Type::Tensor(TensorType::new(elem.unwrap_or(ElemType::F32), shape))
    }

    /// Infers a block expression; the type of the final item wins.
    pub(crate) fn infer_block(&mut self, items: &[apeiron_ast::Item]) -> Type {
        self.push_scope(crate::symbol::ScopeKind::Block);
        let mut result = Type::Unit;
        for item in items {
            result = self.check_item(item);
        }
        self.pop_scope();
        result
    }

    // ── operators ───────────────────────────────────────────────────

    fn infer_unary(&mut self, op: UnaryOp, operand: Type, span: Span) -> Type {
        if operand.is_sentinel() {
            return operand;
        }
        match op {
            UnaryOp::Neg => {
                if operand.is_numeric() || operand.is_tensor() {
                    operand
                } else {
                    self.error(
                        "AN0003",
                        format!("cannot negate a value of type {operand}"),
                        span,
                    );
                    Type::Error
                }
            }
            UnaryOp::Not => {
                if operand == Type::Bool {
                    Type::Bool
                } else {
                    self.error(
                        "AN0003",
                        format!("operator `!` expects a bool, found {operand}"),
                        span,
                    );
                    Type::Error
                }
            }
        }
    }

    fn infer_binary(
        &mut self,
        op: BinaryOp,
        lhs: &Type,
        rhs: &Type,
        lhs_expr: &Expr,
        span: Span,
    ) -> Type {
        match op {
            BinaryOp::Assign => self.infer_assign(lhs_expr, lhs, rhs, span),
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge => {
                if lhs.is_sentinel() || rhs.is_sentinel() {
                    return Type::Bool;
                }
                if !self.comparable(op, lhs, rhs) {
                    self.error("AN0003", format!("cannot compare {lhs} with {rhs}"), span);
                }
                Type::Bool
            }
            BinaryOp::MatMul => self.infer_matmul(lhs, rhs, span),
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => {
                self.infer_arith(op, lhs, rhs, span)
            }
        }
    }

    fn comparable(&self, op: BinaryOp, lhs: &Type, rhs: &Type) -> bool {
        match (lhs, rhs) {
            (Type::Tensor(a), Type::Tensor(b)) => TensorType::broadcast(a, b).is_some(),
            (Type::Str, Type::Str) | (Type::Bool, Type::Bool) => {
                matches!(op, BinaryOp::Eq | BinaryOp::Ne)
            }
            _ => lhs.numeric_elem().is_some() && rhs.numeric_elem().is_some(),
        }
    }

    fn infer_arith(&mut self, op: BinaryOp, lhs: &Type, rhs: &Type, span: Span) -> Type {
        if lhs.is_sentinel() {
            return lhs.clone();
        }
        if rhs.is_sentinel() {
            return rhs.clone();
        }
        match (lhs, rhs) {
            (Type::Tensor(a), Type::Tensor(b)) => match TensorType::broadcast(a, b) {
                Some(result) => rank_to_type(result),
                None => {
                    self.error(
                        "AN0004",
                        format!("shape mismatch for `{op}`: {a} and {b} do not broadcast"),
                        span,
                    );
                    Type::Error
                }
            },
            (Type::Tensor(a), other) | (other, Type::Tensor(a)) => match other.numeric_elem() {
                Some(elem) => match a.elem.unify(elem) {
                    Some(unified) => Type::Tensor(TensorType::new(unified, a.shape.clone())),
                    None => {
                        self.error(
                            "AN0003",
                            format!("cannot apply `{op}` to {lhs} and {rhs}"),
                            span,
                        );
                        Type::Error
                    }
                },
                None => {
                    self.error(
                        "AN0003",
                        format!("cannot apply `{op}` to {lhs} and {rhs}"),
                        span,
                    );
                    Type::Error
                }
            },
            _ => match (lhs.numeric_elem(), rhs.numeric_elem()) {
                (Some(a), Some(b)) => match a.unify(b) {
                    Some(unified) => Type::scalar(unified),
                    None => {
                        self.error(
                            "AN0003",
                            format!("cannot apply `{op}` to {lhs} and {rhs}"),
                            span,
                        );
                        Type::Error
                    }
                },
                _ => {
                    self.error(
                        "AN0003",
                        format!("cannot apply `{op}` to {lhs} and {rhs}"),
                        span,
                    );
                    Type::Error
                }
            },
        }
    }

    fn infer_matmul(&mut self, lhs: &Type, rhs: &Type, span: Span) -> Type {
        if lhs.is_sentinel() {
            return lhs.clone();
        }
        if rhs.is_sentinel() {
            return rhs.clone();
        }
        match (lhs, rhs) {
            (Type::Tensor(a), Type::Tensor(b)) => match TensorType::matmul(a, b) {
                Some(result) => rank_to_type(result),
                None => {
                    self.error(
                        "AN0004",
                        format!("invalid matrix multiplication: {a} @ {b}"),
                        span,
                    );
                    Type::Error
                }
            },
            _ => {
                self.error_with_help(
                    "AN0003",
                    format!("operator `@` expects tensors, found {lhs} and {rhs}"),
                    span,
                    "use `*` for element-wise multiplication",
                );
                Type::Error
            }
        }
    }

    fn infer_assign(&mut self, lhs_expr: &Expr, lhs: &Type, rhs: &Type, span: Span) -> Type {
        if let Expr::Ident(ident) = lhs_expr {
            match self.lookup(&ident.name) {
                Some(symbol) => {
                    let kind = symbol.kind;
                    let declared = symbol.ty.clone();
                    if !kind.is_mutable() {
                        self.error_with_help(
                            "AN0009",
                            format!("cannot assign to {kind} `{}`", ident.name),
                            ident.span,
                            "only variables and parameters are mutable",
                        );
                    } else if !declared.accepts(rhs) {
                        self.error(
                            "AN0003",
                            format!("cannot assign {rhs} to `{}` of type {declared}", ident.name),
                            span,
                        );
                    }
                }
                None => {
                    self.unresolved(&ident.name, ident.span);
                }
            }
        } else if !matches!(lhs_expr, Expr::Index { .. } | Expr::Field { .. }) {
            self.error("AN0009", "invalid assignment target", span);
        }
        let _ = lhs;
        Type::Unit
    }

    // ── calls, indexing, conditionals ───────────────────────────────

    fn infer_call(&mut self, callee: &Expr, args: &[Expr], span: Span) -> Type {
        let arg_types: Vec<Type> = args.iter().map(|arg| self.infer_expr(arg)).collect();
        let Expr::Ident(ident) = callee else {
            let _ = self.infer_expr(callee);
            self.error(
                "AN0007",
                "only named functions can be called",
                callee.span(),
            );
            return Type::Error;
        };
        let found = self.lookup(&ident.name).map(|symbol| symbol.ty.clone());
        match found {
            Some(Type::Fn { params, ret }) => {
                if params.len() != args.len() {
                    self.error(
                        "AN0007",
                        format!(
                            "function `{}` expects {} argument(s), found {}",
                            ident.name,
                            params.len(),
                            args.len()
                        ),
                        span,
                    );
                } else {
                    for (index, (expected, actual)) in params.iter().zip(&arg_types).enumerate() {
                        if !expected.accepts(actual) {
                            self.error(
                                "AN0003",
                                format!(
                                    "argument {} of `{}`: expected {expected}, found {actual}",
                                    index + 1,
                                    ident.name
                                ),
                                args[index].span(),
                            );
                        }
                    }
                }
                *ret
            }
            Some(other) => {
                self.error(
                    "AN0007",
                    format!("`{}` has type {other} and is not callable", ident.name),
                    ident.span,
                );
                Type::Error
            }
            None => {
                if let Some(result) = self.infer_variant_constructor(&ident.name, &arg_types, span)
                {
                    return result;
                }
                if let Some(signature) = apeiron_ast::builtins::find(&ident.name) {
                    if signature.params.len() != args.len() {
                        self.error(
                            "AN0007",
                            format!(
                                "builtin `{}` expects {} argument(s), found {}",
                                ident.name,
                                signature.params.len(),
                                args.len()
                            ),
                            span,
                        );
                        return Type::Error;
                    }
                    for (index, (expected, actual)) in
                        signature.params.iter().zip(&arg_types).enumerate()
                    {
                        let expected = builtin_type(*expected);
                        if !expected.accepts(actual) {
                            self.error(
                                "AN0003",
                                format!(
                                    "argument {} of `{}`: expected {expected}, found {actual}",
                                    index + 1,
                                    ident.name
                                ),
                                args[index].span(),
                            );
                        }
                    }
                    return builtin_type(signature.ret);
                }
                if let Some(sig) = self.extern_signature(&ident.name).cloned() {
                    return self.infer_extern_call(&ident.name, &sig, &arg_types, args, span);
                }
                self.unresolved(&ident.name, ident.span);
                Type::Error
            }
        }
    }

    /// Type-checks a call to a declared native extern.
    fn infer_extern_call(
        &mut self,
        name: &str,
        sig: &apeiron_core::ExternSig,
        arg_types: &[Type],
        args: &[Expr],
        span: Span,
    ) -> Type {
        if sig.params.len() != args.len() {
            self.error(
                "AN0007",
                format!(
                    "extern `{name}` expects {} argument(s), found {}",
                    sig.params.len(),
                    args.len()
                ),
                span,
            );
            return Type::Error;
        }
        for (index, (expected, actual)) in sig.params.iter().zip(arg_types).enumerate() {
            let expected = extern_type(*expected);
            if !expected.accepts(actual) {
                self.error(
                    "AN0003",
                    format!(
                        "argument {} of `{name}`: expected {expected}, found {actual}",
                        index + 1
                    ),
                    args[index].span(),
                );
            }
        }
        extern_type(sig.ret)
    }

    /// Recognises the `Some`/`Ok`/`Err` enum constructors. Returns `None`
    /// when `name` is not a constructor, so normal resolution continues.
    fn infer_variant_constructor(
        &mut self,
        name: &str,
        arg_types: &[Type],
        span: Span,
    ) -> Option<Type> {
        let variant = match name {
            "Some" | "Ok" | "Err" => name,
            _ => return None,
        };
        if arg_types.len() != 1 {
            self.error(
                "AN0007",
                format!(
                    "`{variant}` expects one argument, found {}",
                    arg_types.len()
                ),
                span,
            );
            return Some(Type::Error);
        }
        let payload = arg_types[0].clone();
        Some(match variant {
            "Some" => Type::Option(Box::new(payload)),
            "Ok" => Type::Result(Box::new(payload), Box::new(Type::Unknown)),
            _ => Type::Result(Box::new(Type::Unknown), Box::new(payload)),
        })
    }

    fn infer_index(&mut self, target: &Expr, index: &Expr, span: Span) -> Type {
        let target_ty = self.infer_expr(target);
        let index_ty = self.infer_expr(index);
        if !index_ty.is_sentinel() && index_ty != Type::I64 {
            self.error(
                "AN0003",
                format!("index must be i64, found {index_ty}"),
                index.span(),
            );
        }
        match target_ty {
            Type::Tensor(tensor) if tensor.rank() == 0 => {
                self.error("AN0004", "cannot index a scalar value", span);
                Type::Error
            }
            Type::Tensor(tensor) => {
                if tensor.rank() == 1 {
                    Type::scalar(tensor.elem)
                } else {
                    Type::Tensor(TensorType::new(tensor.elem, tensor.shape[1..].to_vec()))
                }
            }
            Type::Error => Type::Error,
            Type::Unknown => Type::Unknown,
            other => {
                self.error(
                    "AN0004",
                    format!("cannot index a value of type {other}"),
                    span,
                );
                Type::Error
            }
        }
    }

    fn infer_if(
        &mut self,
        condition: &Expr,
        then_branch: &Expr,
        else_branch: Option<&Expr>,
        span: Span,
    ) -> Type {
        let condition_ty = self.infer_expr(condition);
        if !condition_ty.is_sentinel() && condition_ty != Type::Bool {
            self.error(
                "AN0003",
                format!("`if` condition must be bool, found {condition_ty}"),
                condition.span(),
            );
        }
        let then_ty = self.infer_expr(then_branch);
        match else_branch {
            Some(else_branch) => {
                let else_ty = self.infer_expr(else_branch);
                self.unify_branches(&then_ty, &else_ty, span)
            }
            None => Type::Unit,
        }
    }

    fn unify_branches(&mut self, then_ty: &Type, else_ty: &Type, span: Span) -> Type {
        self.unify_branches_as(then_ty, else_ty, span, "`if` branches")
    }

    /// Unifies two branch types, reporting `label` in the diagnostic so
    /// `if` and `match` can share the same lattice rules.
    fn unify_branches_as(
        &mut self,
        then_ty: &Type,
        else_ty: &Type,
        span: Span,
        label: &str,
    ) -> Type {
        if then_ty == else_ty {
            return then_ty.clone();
        }
        if then_ty.is_sentinel() {
            return else_ty.clone();
        }
        if else_ty.is_sentinel() {
            return then_ty.clone();
        }
        let message = format!("{label} have incompatible types {then_ty} and {else_ty}");
        match (then_ty, else_ty) {
            (Type::Tensor(a), Type::Tensor(b)) => match TensorType::broadcast(a, b) {
                Some(result) => rank_to_type(result),
                None => {
                    self.error("AN0003", message, span);
                    Type::Error
                }
            },
            _ => match (then_ty.numeric_elem(), else_ty.numeric_elem()) {
                (Some(a), Some(b)) => match a.unify(b) {
                    Some(unified) => Type::scalar(unified),
                    None => {
                        self.error("AN0003", message, span);
                        Type::Error
                    }
                },
                _ => {
                    // Allow enum branches to unify if one has more specific
                    // payload types (e.g. `Some(x)` gives Option[T], `None`
                    // gives Option[Unknown]); accept both.
                    if then_ty.accepts(else_ty) || else_ty.accepts(then_ty) {
                        return then_ty.clone();
                    }
                    self.error("AN0003", message, span);
                    Type::Error
                }
            },
        }
    }

    // ── match ───────────────────────────────────────────────────────

    /// Types a `match` expression: every arm is checked in its own scope
    /// with the pattern bindings in scope, and arm bodies are unified.
    fn infer_match(&mut self, scrutinee: &Expr, arms: &[apeiron_ast::Arm], span: Span) -> Type {
        let scrutinee_ty = self.infer_expr(scrutinee);
        if arms.is_empty() {
            self.error("AN0017", "`match` must have at least one arm", span);
            return Type::Unit;
        }

        let mut result: Option<Type> = None;
        let mut covered = false;
        for arm in arms {
            if covered {
                self.error("AN0018", "this `match` arm is unreachable", arm.span);
            }
            self.push_scope(crate::symbol::ScopeKind::Block);
            self.bind_pattern(&arm.pat, &scrutinee_ty, scrutinee.span());
            if let Some(guard) = &arm.guard {
                let guard_ty = self.infer_expr(guard);
                if !guard_ty.is_sentinel() && guard_ty != Type::Bool {
                    self.error(
                        "AN0003",
                        format!("`match` guard must be bool, found {guard_ty}"),
                        guard.span(),
                    );
                }
            } else if arm.pat.is_irrefutable() {
                covered = true;
            }
            let body_ty = self.infer_expr(&arm.body);
            self.pop_scope();
            result = Some(match result {
                None => body_ty,
                Some(accumulated) => {
                    self.unify_branches_as(&accumulated, &body_ty, span, "`match` arms")
                }
            });
        }

        if !covered && !self.match_is_exhaustive(&scrutinee_ty, arms) {
            self.error_with_help(
                "AN0017",
                "non-exhaustive `match`: no arm covers every value",
                span,
                "add a final `_ => ...` arm",
            );
            return Type::Error;
        }

        result.unwrap_or(Type::Unit)
    }

    /// Whether the arms cover every value of `scrutinee_ty`.
    ///
    /// `bool`, `Option` and `Result` are the only types with a finite set
    /// of constructors the analyzer can enumerate.
    fn match_is_exhaustive(&self, scrutinee_ty: &Type, arms: &[apeiron_ast::Arm]) -> bool {
        let names: Vec<&str> = arms
            .iter()
            .filter(|arm| arm.guard.is_none())
            .filter_map(|arm| match &arm.pat {
                Pattern::Variant { name, .. } => Some(name.name.as_str()),
                _ => None,
            })
            .collect();
        match scrutinee_ty {
            Type::Bool => {
                let mut seen_true = false;
                let mut seen_false = false;
                for arm in arms {
                    if arm.guard.is_some() {
                        continue;
                    }
                    match &arm.pat {
                        Pattern::Literal(Literal::Bool(true)) => seen_true = true,
                        Pattern::Literal(Literal::Bool(false)) => seen_false = true,
                        _ => {}
                    }
                }
                seen_true && seen_false
            }
            Type::Option(_) => names.contains(&"Some") && names.contains(&"None"),
            Type::Result(_, _) => names.contains(&"Ok") && names.contains(&"Err"),
            _ => false,
        }
    }

    /// Declares the names bound by `pat` and checks it against `scrutinee`.
    fn bind_pattern(&mut self, pat: &Pattern, scrutinee: &Type, span: Span) {
        match pat {
            Pattern::Wildcard => {}
            Pattern::Bind(ident) => {
                self.declare(crate::symbol::Symbol::new(
                    ident.name.clone(),
                    crate::symbol::SymbolKind::Variable,
                    scrutinee.clone(),
                    ident.span,
                ));
            }
            Pattern::Literal(literal) => {
                if scrutinee.is_sentinel() {
                    return;
                }
                let pattern_ty = literal_type(literal);
                if self.unify_pattern_types(&pattern_ty, scrutinee) {
                    return;
                }
                self.error(
                    "AN0018",
                    format!("pattern of type {pattern_ty} cannot match {scrutinee}"),
                    span,
                );
            }
            Pattern::Variant { name, inner } => {
                self.bind_variant_pattern(name, inner.as_deref(), scrutinee, span);
            }
            Pattern::Tuple(elements) => {
                if scrutinee.is_sentinel() {
                    return;
                }
                let Type::Tensor(tensor) = scrutinee else {
                    self.error(
                        "AN0018",
                        format!("tuple pattern cannot match {scrutinee}"),
                        span,
                    );
                    return;
                };
                let (first, rest) = tensor.shape.split_first().expect("tensor shape");
                match first {
                    Some(count) if *count as usize != elements.len() => {
                        self.error(
                            "AN0018",
                            format!(
                                "pattern has {} elements but the value has {count}",
                                elements.len()
                            ),
                            span,
                        );
                        return;
                    }
                    _ => {}
                }
                let element_ty = if rest.is_empty() {
                    Type::scalar(tensor.elem)
                } else {
                    Type::Tensor(TensorType::new(tensor.elem, rest.to_vec()))
                };
                for element in elements {
                    self.bind_pattern(element, &element_ty, span);
                }
            }
        }
    }

    /// Whether a literal of type `pattern_ty` can match `scrutinee`.
    fn unify_pattern_types(&self, pattern_ty: &Type, scrutinee: &Type) -> bool {
        if pattern_ty == scrutinee {
            return true;
        }
        match (pattern_ty.numeric_elem(), scrutinee.numeric_elem()) {
            (Some(a), Some(b)) => a.unify(b).is_some(),
            _ => false,
        }
    }

    /// Checks an `Option`/`Result` variant pattern and binds its payload.
    fn bind_variant_pattern(
        &mut self,
        name: &Ident,
        inner: Option<&Pattern>,
        scrutinee: &Type,
        span: Span,
    ) {
        let variant = name.name.as_str();
        if scrutinee.is_sentinel() {
            if let Some(inner) = inner {
                self.bind_pattern(inner, &Type::Unknown, span);
            }
            return;
        }
        let payload = match (variant, scrutinee) {
            ("Some", Type::Option(ok)) => Some((**ok).clone()),
            ("Ok", Type::Result(ok, _)) => Some((**ok).clone()),
            ("Err", Type::Result(_, err)) => Some((**err).clone()),
            ("None", Type::Option(_)) => None,
            _ => {
                self.error(
                    "AN0018",
                    format!("variant `{}` cannot match {scrutinee}", name.name),
                    span,
                );
                if let Some(inner) = inner {
                    self.bind_pattern(inner, &Type::Unknown, span);
                }
                return;
            }
        };
        match (inner, payload) {
            (Some(inner), Some(payload)) => self.bind_pattern(inner, &payload, span),
            (None, None) => {}
            (Some(_), None) => self.error(
                "AN0018",
                format!("variant `{}` carries no payload", name.name),
                span,
            ),
            (None, Some(_)) => self.error(
                "AN0018",
                format!("variant `{}` requires a payload pattern", name.name),
                span,
            ),
        }
    }

    /// Types the postfix `?` operator and checks it against the enclosing
    /// function's declared return type.
    fn infer_try(&mut self, operand: &Expr, span: Span) -> Type {
        let operand_ty = self.infer_expr(operand);
        let return_ty = self.function_return.clone();
        match operand_ty {
            Type::Option(inner) => {
                self.check_try_return(
                    &return_ty,
                    |ty| matches!(ty, Type::Option(_)),
                    "Option",
                    span,
                );
                *inner
            }
            Type::Result(ok, err) => {
                if let Some(Type::Result(_, expected_err)) = &return_ty {
                    if !expected_err.accepts(&err) {
                        self.error(
                            "AN0003",
                            format!(
                                "`?` propagates an error of type {err} but the function returns {expected_err}"
                            ),
                            span,
                        );
                    }
                } else {
                    self.check_try_return(
                        &return_ty,
                        |ty| matches!(ty, Type::Result(_, _)),
                        "Result",
                        span,
                    );
                }
                *ok
            }
            Type::Unknown | Type::Error => operand_ty,
            other => {
                self.error(
                    "AN0003",
                    format!("`?` requires an Option or Result, found {other}"),
                    span,
                );
                Type::Error
            }
        }
    }

    /// Reports when `?` is used where the function cannot propagate.
    fn check_try_return(
        &mut self,
        return_ty: &Option<Type>,
        accepts: impl Fn(&Type) -> bool,
        expected: &str,
        span: Span,
    ) {
        match return_ty {
            None => self.error(
                "AN0003",
                format!("`?` can only be used inside a function returning {expected}"),
                span,
            ),
            Some(ty) if !ty.is_sentinel() && !accepts(ty) => self.error(
                "AN0003",
                format!("`?` on {expected} requires the function to return {expected}, found {ty}"),
                span,
            ),
            Some(_) => {}
        }
    }

    // ── explicit type syntax ────────────────────────────────────────

    /// Resolves a written type annotation.
    pub(crate) fn type_from_typeexpr(&mut self, ty: &TypeExpr) -> Type {
        let name = ty.name.name.as_str();
        match name {
            "Option" => {
                if ty.dims.len() != 1 {
                    self.error(
                        "AN0006",
                        format!(
                            "`Option` expects one type argument, found {}",
                            ty.dims.len()
                        ),
                        ty.span,
                    );
                    return Type::Error;
                }
                return Type::Option(Box::new(self.type_from_dim(&ty.dims[0])));
            }
            "Result" => {
                if ty.dims.len() != 2 {
                    self.error(
                        "AN0006",
                        format!(
                            "`Result` expects two type arguments, found {}",
                            ty.dims.len()
                        ),
                        ty.span,
                    );
                    return Type::Error;
                }
                let ok = self.type_from_dim(&ty.dims[0]);
                let err = self.type_from_dim(&ty.dims[1]);
                return Type::Result(Box::new(ok), Box::new(err));
            }
            _ => {}
        }
        if !ty.dims.is_empty() {
            let Some(elem) = numeric_elem_for(name) else {
                self.error(
                    "AN0006",
                    format!("`{name}` cannot be used as a tensor element type"),
                    ty.name.span,
                );
                return Type::Error;
            };
            return Type::Tensor(TensorType::new(elem, dims_to_shape(&ty.dims)));
        }
        self.type_from_name(name, ty.name.span)
    }

    /// Resolves a bare (unshaped) type name to an analyzer type.
    fn type_from_name(&mut self, name: &str, span: Span) -> Type {
        if let Some(elem) = numeric_elem_for(name) {
            return Type::scalar(elem);
        }
        match name {
            "bool" => Type::Bool,
            "str" | "string" => Type::Str,
            "unit" | "()" => Type::Unit,
            _ => {
                self.error("AN0006", format!("unknown type `{name}`"), span);
                Type::Error
            }
        }
    }

    /// Resolves one generic argument such as the `i64` in `Option[i64]`.
    fn type_from_dim(&mut self, dim: &TypeDim) -> Type {
        match dim {
            TypeDim::Dynamic(ident) => self.type_from_name(&ident.name, ident.span),
            TypeDim::Const { span, .. } => {
                self.error("AN0006", "expected a type name", *span);
                Type::Error
            }
        }
    }
}

fn literal_type(literal: &Literal) -> Type {
    match literal {
        Literal::Int(_) => Type::I64,
        Literal::Float(_) => Type::F32,
        Literal::Str(_) => Type::Str,
        Literal::Bool(_) => Type::Bool,
    }
}

/// Maps a built-in signature type onto the analyzer's type lattice. Tensors
/// carry no declared shape, so they map to [`Type::Unknown`] and pass any
/// tensor-shaped argument without triggering spurious shape diagnostics.
fn builtin_type(ret: BuiltinRet) -> Type {
    match ret {
        BuiltinRet::Unit => Type::Unit,
        BuiltinRet::I64 => Type::I64,
        BuiltinRet::F32 => Type::F32,
        BuiltinRet::F64 => Type::F64,
        BuiltinRet::Bool => Type::Bool,
        BuiltinRet::Tensor => Type::Unknown,
    }
}

/// Maps a native extern's C ABI type onto the analyzer's type lattice.
fn extern_type(abi: AbiType) -> Type {
    match abi {
        AbiType::F64 => Type::F64,
        AbiType::I64 => Type::I64,
    }
}

fn numeric_elem_for(name: &str) -> Option<ElemType> {
    match name {
        "i8" | "i16" | "i32" | "i64" | "int" => Some(ElemType::I64),
        "f16" | "f32" | "float" => Some(ElemType::F32),
        "f64" | "double" => Some(ElemType::F64),
        _ => None,
    }
}

fn dims_to_shape(dims: &[TypeDim]) -> Vec<Option<u64>> {
    dims.iter()
        .map(|dim| match dim {
            TypeDim::Const { value, .. } => Some(*value),
            TypeDim::Dynamic(_) => None,
        })
        .collect()
}

/// Converts a rank-0 tensor back into a scalar type.
fn rank_to_type(tensor: TensorType) -> Type {
    if tensor.rank() == 0 {
        Type::scalar(tensor.elem)
    } else {
        Type::Tensor(tensor)
    }
}

fn format_shape(shape: &[Option<u64>]) -> String {
    let dims: Vec<String> = shape
        .iter()
        .map(|dim| match dim {
            Some(value) => value.to_string(),
            None => "_".to_string(),
        })
        .collect();
    format!("[{}]", dims.join(", "))
}

/// Convenience helper used by tests and the checker.
pub(crate) fn is_text(ty: &Type) -> bool {
    matches!(ty, Type::Str) || ty.is_sentinel()
}

/// Convenience helper used by tests and the checker.
pub(crate) fn is_number(ty: &Type) -> bool {
    ty.numeric_elem().is_some() || ty.is_sentinel()
}

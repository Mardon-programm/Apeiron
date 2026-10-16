//! Expression type inference and tensor shape inference.

use apeiron_ast::builtins::BuiltinRet;
use apeiron_ast::{BinaryOp, Expr, Literal, TypeDim, TypeExpr, UnaryOp};
use apeiron_core::span::Span;

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
                        self.unresolved(&ident.name, ident.span);
                        Type::Error
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
                if !self.comparable(lhs, rhs) {
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

    fn comparable(&self, lhs: &Type, rhs: &Type) -> bool {
        match (lhs, rhs) {
            (Type::Tensor(a), Type::Tensor(b)) => TensorType::broadcast(a, b).is_some(),
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
                self.unresolved(&ident.name, ident.span);
                Type::Error
            }
        }
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
        if then_ty == else_ty {
            return then_ty.clone();
        }
        if then_ty.is_sentinel() {
            return else_ty.clone();
        }
        if else_ty.is_sentinel() {
            return then_ty.clone();
        }
        match (then_ty, else_ty) {
            (Type::Tensor(a), Type::Tensor(b)) => match TensorType::broadcast(a, b) {
                Some(result) => rank_to_type(result),
                None => {
                    self.error(
                        "AN0003",
                        format!("`if` branches have incompatible types {then_ty} and {else_ty}"),
                        span,
                    );
                    Type::Error
                }
            },
            _ => match (then_ty.numeric_elem(), else_ty.numeric_elem()) {
                (Some(a), Some(b)) => match a.unify(b) {
                    Some(unified) => Type::scalar(unified),
                    None => {
                        self.error(
                            "AN0003",
                            format!(
                                "`if` branches have incompatible types {then_ty} and {else_ty}"
                            ),
                            span,
                        );
                        Type::Error
                    }
                },
                _ => {
                    self.error(
                        "AN0003",
                        format!("`if` branches have incompatible types {then_ty} and {else_ty}"),
                        span,
                    );
                    Type::Error
                }
            },
        }
    }

    // ── explicit type syntax ────────────────────────────────────────

    /// Resolves a written type annotation.
    pub(crate) fn type_from_typeexpr(&mut self, ty: &TypeExpr) -> Type {
        let name = ty.name.name.as_str();
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
        if let Some(elem) = numeric_elem_for(name) {
            return Type::scalar(elem);
        }
        match name {
            "bool" => Type::Bool,
            "str" | "string" => Type::Str,
            "unit" | "()" => Type::Unit,
            _ => {
                self.error("AN0006", format!("unknown type `{name}`"), ty.name.span);
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

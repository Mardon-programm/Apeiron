//! Runtime values.
//!
//! Scalars are the primitive machine values; tensors are dense, row-major,
//! and always numeric. Element types are preserved for display and for
//! deciding whether integer arithmetic should stay integral.

use std::fmt;

use apeiron_ast::{BinaryOp, UnaryOp};
use apeiron_ir::{ConstValue, ElemType, TensorLit};

use crate::error::{Result, RuntimeError};

/// A dense, row-major numeric tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Element type.
    pub elem: ElemType,
    /// Shape, outermost axis first.
    pub shape: Vec<usize>,
    /// Flat data.
    pub data: Vec<f64>,
}

impl Tensor {
    /// Builds a tensor from a shape and flat data.
    pub fn new(elem: ElemType, shape: Vec<usize>, data: Vec<f64>) -> Self {
        Self { elem, shape, data }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// A rank-0 (scalar) tensor.
    pub fn scalar(elem: ElemType, value: f64) -> Self {
        Self {
            elem,
            shape: Vec::new(),
            data: vec![value],
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims = self
            .shape
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "{}[{}] {}", self.elem, dims, format_data(&self.data))
    }
}

fn format_data(data: &[f64]) -> String {
    let items = data
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{items}]")
}

/// A value produced or consumed by the interpreter.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// The unit value.
    Unit,
    /// A 64-bit integer.
    Int(i64),
    /// A floating point number.
    Float(f64),
    /// A boolean.
    Bool(bool),
    /// A string.
    Str(String),
    /// A tensor.
    Tensor(Tensor),
    /// An `Option`/`Result` variant.
    Enum(EnumValue),
}

/// The payload of an `Option`/`Result` value.
#[derive(Clone, Debug, PartialEq)]
pub struct EnumValue {
    /// Enum type name (`Option` or `Result`).
    pub type_name: String,
    /// Constructor name (`Some`, `None`, `Ok`, `Err`).
    pub variant: String,
    /// Payload values, in constructor argument order.
    pub payload: Vec<Value>,
}

impl EnumValue {
    /// Builds an enum value.
    pub fn new(
        type_name: impl Into<String>,
        variant: impl Into<String>,
        payload: Vec<Value>,
    ) -> Self {
        Self {
            type_name: type_name.into(),
            variant: variant.into(),
            payload,
        }
    }

    /// Whether this value carries the named variant.
    pub fn is(&self, variant: &str) -> bool {
        self.variant == variant
    }
}

impl fmt::Display for EnumValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.variant)?;
        if !self.payload.is_empty() {
            f.write_str("(")?;
            for (index, value) in self.payload.iter().enumerate() {
                if index > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{value}")?;
            }
            f.write_str(")")?;
        }
        Ok(())
    }
}

impl Value {
    /// Materialises a compile-time constant.
    pub fn from_const(value: &ConstValue) -> Self {
        match value {
            ConstValue::Int(inner) => Value::Int(*inner),
            ConstValue::Float(inner) => Value::Float(*inner),
            ConstValue::Bool(inner) => Value::Bool(*inner),
            ConstValue::Str(inner) => Value::Str(inner.clone()),
            ConstValue::Unit => Value::Unit,
        }
    }

    /// Materialises a static tensor literal.
    pub fn from_tensor_lit(literal: &TensorLit) -> Self {
        Value::Tensor(Tensor::new(
            literal.elem,
            literal.shape.iter().map(|dim| *dim as usize).collect(),
            literal.data.clone(),
        ))
    }

    /// Interprets the value as a float, when numeric.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(inner) => Some(*inner as f64),
            Value::Float(inner) => Some(*inner),
            Value::Tensor(tensor) if tensor.data.len() == 1 => Some(tensor.data[0]),
            _ => None,
        }
    }

    /// Interprets the value as an integer, when numeric.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(inner) => Some(*inner),
            Value::Float(inner) => Some(*inner as i64),
            Value::Tensor(tensor) if tensor.data.len() == 1 => Some(tensor.data[0] as i64),
            _ => None,
        }
    }

    /// Interprets the value as a boolean, when logical.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(inner) => Some(*inner),
            _ => None,
        }
    }

    /// Interprets the value as text, when textual.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(inner) => Some(inner.as_str()),
            _ => None,
        }
    }

    /// Truthiness for conditionals: numbers are non-zero, tensors non-empty.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(inner) => *inner,
            Value::Int(inner) => *inner != 0,
            Value::Float(inner) => *inner != 0.0,
            Value::Str(inner) => !inner.is_empty(),
            Value::Tensor(tensor) => !tensor.data.is_empty(),
            Value::Enum(_) => true,
            Value::Unit => false,
        }
    }

    /// Whether the value is a tensor.
    pub fn is_tensor(&self) -> bool {
        matches!(self, Value::Tensor(_))
    }

    /// Whether the value is a scalar number (`Int` or `Float`).
    pub fn is_numeric_scalar(&self) -> bool {
        matches!(self, Value::Int(_) | Value::Float(_))
    }

    /// Shape of a tensor value, if any.
    pub fn shape(&self) -> Option<&[usize]> {
        match self {
            Value::Tensor(tensor) => Some(&tensor.shape),
            _ => None,
        }
    }

    /// A short type name for diagnostics.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Unit => "unit",
            Value::Int(_) => "i64",
            Value::Float(_) => "f64",
            Value::Bool(_) => "bool",
            Value::Str(_) => "str",
            Value::Tensor(_) => "tensor",
            Value::Enum(enum_value) => match enum_value.type_name.as_str() {
                "Result" => "Result",
                _ => "Option",
            },
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => f.write_str("()"),
            Value::Int(inner) => write!(f, "{inner}"),
            Value::Float(inner) => write!(f, "{inner}"),
            Value::Bool(inner) => write!(f, "{inner}"),
            Value::Str(inner) => write!(f, "{inner}"),
            Value::Tensor(tensor) => write!(f, "{tensor}"),
            Value::Enum(enum_value) => write!(f, "{enum_value}"),
        }
    }
}

/// Applies a prefix operator.
pub fn unary(op: UnaryOp, operand: &Value) -> Result<Value> {
    match op {
        UnaryOp::Neg => match operand {
            Value::Int(inner) => Ok(Value::Int(-inner)),
            Value::Float(inner) => Ok(Value::Float(-inner)),
            Value::Tensor(tensor) => Ok(Value::Tensor(Tensor::new(
                tensor.elem,
                tensor.shape.clone(),
                tensor.data.iter().map(|value| -value).collect(),
            ))),
            other => Err(RuntimeError::Type(format!(
                "cannot negate {}",
                other.type_name()
            ))),
        },
        UnaryOp::Not => Ok(Value::Bool(!operand.is_truthy())),
    }
}

/// Applies an infix operator.
pub fn binary(op: BinaryOp, lhs: &Value, rhs: &Value) -> Result<Value> {
    match (lhs, rhs) {
        (Value::Tensor(left), Value::Tensor(right)) => tensor_binary(op, left, right),
        (Value::Tensor(tensor), scalar) if scalar.is_numeric_scalar() => {
            tensor_scalar(op, tensor, scalar, false)
        }
        (scalar, Value::Tensor(tensor)) if scalar.is_numeric_scalar() => {
            tensor_scalar(op, tensor, scalar, true)
        }
        _ => scalar_binary(op, lhs, rhs),
    }
}

fn scalar_binary(op: BinaryOp, lhs: &Value, rhs: &Value) -> Result<Value> {
    if op.is_comparison() {
        if let (Value::Str(a), Value::Str(b)) = (lhs, rhs) {
            return str_compare(op, a, b);
        }
        if let (Value::Bool(a), Value::Bool(b)) = (lhs, rhs) {
            return bool_compare(op, *a, *b);
        }
        if let (Value::Enum(a), Value::Enum(b)) = (lhs, rhs) {
            return match op {
                BinaryOp::Eq => Ok(Value::Bool(a == b)),
                BinaryOp::Ne => Ok(Value::Bool(a != b)),
                other => Err(RuntimeError::Type(format!(
                    "cannot order enum values with {other}"
                ))),
            };
        }
        let (a, b) = (lhs.as_f64(), rhs.as_f64());
        let (Some(a), Some(b)) = (a, b) else {
            return Err(RuntimeError::Type(format!(
                "cannot compare {} with {}",
                lhs.type_name(),
                rhs.type_name()
            )));
        };
        return Ok(Value::Bool(compare(op, a, b)));
    }

    let float = matches!(lhs, Value::Float(_)) || matches!(rhs, Value::Float(_));
    match op {
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => {}
        other => {
            return Err(RuntimeError::Type(format!(
                "unsupported operator {other} on {}/{}",
                lhs.type_name(),
                rhs.type_name()
            )))
        }
    }

    if !float {
        if let (Some(a), Some(b)) = (lhs.as_i64(), rhs.as_i64()) {
            return int_arith(op, a, b);
        }
    }

    let (Some(a), Some(b)) = (lhs.as_f64(), rhs.as_f64()) else {
        return Err(RuntimeError::Type(format!(
            "cannot apply {op} to {} and {}",
            lhs.type_name(),
            rhs.type_name()
        )));
    };
    float_arith(op, a, b)
}

fn int_arith(op: BinaryOp, a: i64, b: i64) -> Result<Value> {
    match op {
        BinaryOp::Add => Ok(Value::Int(a + b)),
        BinaryOp::Sub => Ok(Value::Int(a - b)),
        BinaryOp::Mul => Ok(Value::Int(a * b)),
        BinaryOp::Div => {
            if b == 0 {
                Err(RuntimeError::Arithmetic("integer division by zero".into()))
            } else {
                Ok(Value::Int(a / b))
            }
        }
        BinaryOp::Rem => {
            if b == 0 {
                Err(RuntimeError::Arithmetic("integer remainder by zero".into()))
            } else {
                Ok(Value::Int(a % b))
            }
        }
        _ => Err(RuntimeError::Type(format!(
            "{op} is not integer arithmetic"
        ))),
    }
}

fn float_arith(op: BinaryOp, a: f64, b: f64) -> Result<Value> {
    float_number(op, a, b).map(Value::Float)
}

fn float_number(op: BinaryOp, a: f64, b: f64) -> Result<f64> {
    match op {
        BinaryOp::Add => Ok(a + b),
        BinaryOp::Sub => Ok(a - b),
        BinaryOp::Mul => Ok(a * b),
        BinaryOp::Div => Ok(a / b),
        BinaryOp::Rem => Ok(a % b),
        _ => Err(RuntimeError::Type(format!("{op} is not float arithmetic"))),
    }
}

fn compare(op: BinaryOp, a: f64, b: f64) -> bool {
    match op {
        BinaryOp::Eq => a == b,
        BinaryOp::Ne => a != b,
        BinaryOp::Lt => a < b,
        BinaryOp::Le => a <= b,
        BinaryOp::Gt => a > b,
        BinaryOp::Ge => a >= b,
        _ => false,
    }
}

/// Equality for string values; ordering operators are rejected.
fn str_compare(op: BinaryOp, a: &str, b: &str) -> Result<Value> {
    match op {
        BinaryOp::Eq => Ok(Value::Bool(a == b)),
        BinaryOp::Ne => Ok(Value::Bool(a != b)),
        other => Err(RuntimeError::Type(format!(
            "cannot order str values with {other}"
        ))),
    }
}

/// Equality for boolean values; ordering operators are rejected.
fn bool_compare(op: BinaryOp, a: bool, b: bool) -> Result<Value> {
    match op {
        BinaryOp::Eq => Ok(Value::Bool(a == b)),
        BinaryOp::Ne => Ok(Value::Bool(a != b)),
        other => Err(RuntimeError::Type(format!(
            "cannot order bool values with {other}"
        ))),
    }
}

fn tensor_binary(op: BinaryOp, left: &Tensor, right: &Tensor) -> Result<Value> {
    let (shape, pairs) = broadcast(left, right)?;
    if op.is_comparison() {
        let result = pairs.iter().all(|(a, b)| compare(op, *a, *b));
        return Ok(Value::Bool(result));
    }
    let elem = unify_elem(left.elem, right.elem);
    let data = pairs
        .iter()
        .map(|(a, b)| float_number(op, *a, *b))
        .collect::<Result<Vec<_>>>()?;
    Ok(Value::Tensor(Tensor::new(elem, shape, data)))
}

/// Applies an operator between a tensor and a scalar number.
fn tensor_scalar(
    op: BinaryOp,
    tensor: &Tensor,
    scalar: &Value,
    scalar_on_left: bool,
) -> Result<Value> {
    let scalar_elem = if matches!(scalar, Value::Float(_)) {
        ElemType::F64
    } else {
        ElemType::I64
    };
    let scalar = scalar
        .as_f64()
        .expect("a numeric scalar always converts to f64");
    if op.is_comparison() {
        let result = tensor.data.iter().all(|value| {
            let (a, b) = if scalar_on_left {
                (scalar, *value)
            } else {
                (*value, scalar)
            };
            compare(op, a, b)
        });
        return Ok(Value::Bool(result));
    }
    let data = tensor
        .data
        .iter()
        .map(|value| {
            let (a, b) = if scalar_on_left {
                (scalar, *value)
            } else {
                (*value, scalar)
            };
            float_number(op, a, b)
        })
        .collect::<Result<Vec<_>>>()?;
    let elem = unify_elem(tensor.elem, scalar_elem);
    Ok(Value::Tensor(Tensor::new(elem, tensor.shape.clone(), data)))
}

/// The numeric element type that can hold both operands.
fn unify_elem(left: ElemType, right: ElemType) -> ElemType {
    match (left, right) {
        (ElemType::F64, _) | (_, ElemType::F64) => ElemType::F64,
        (ElemType::F32, _) | (_, ElemType::F32) => ElemType::F32,
        _ => ElemType::I64,
    }
}

/// A common shape together with its per-element operand pairs.
type Broadcast = (Vec<usize>, Vec<(f64, f64)>);

/// Broadcasts two tensors to a common shape, returning per-element pairs.
fn broadcast(left: &Tensor, right: &Tensor) -> Result<Broadcast> {
    let shape = broadcast_shape(&left.shape, &right.shape)?;
    let left_offsets = broadcast_offsets(&left.shape, &shape);
    let right_offsets = broadcast_offsets(&right.shape, &shape);
    let pairs = left_offsets
        .into_iter()
        .zip(right_offsets)
        .map(|(l, r)| (left.data[l], right.data[r]))
        .collect();
    Ok((shape, pairs))
}

/// The NumPy-style broadcast of two concrete shapes.
fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    let rank = left.len().max(right.len());
    let mut shape = vec![1usize; rank];
    for position in 0..rank {
        let a = axis_from_right(left, position);
        let b = axis_from_right(right, position);
        let dim = match (a, b) {
            (Some(x), Some(y)) if x == y => x,
            (Some(1), Some(y)) => y,
            (Some(x), Some(1)) => x,
            (Some(x), None) | (None, Some(x)) => x,
            (Some(x), Some(y)) => {
                return Err(RuntimeError::Arithmetic(format!(
                    "shape mismatch: {:?} vs {:?} (axis {x} vs {y})",
                    left, right
                )))
            }
            (None, None) => unreachable!("position is below the broadcast rank"),
        };
        shape[rank - 1 - position] = dim;
    }
    Ok(shape)
}

/// The `position`-th axis counting from the right, when present.
fn axis_from_right(shape: &[usize], position: usize) -> Option<usize> {
    shape.len().checked_sub(position + 1).map(|i| shape[i])
}

/// Maps each flat index of `out` to the flat index it reads in `src`.
///
/// Dimensions that `src` lacks or that have extent one are broadcast.
fn broadcast_offsets(src: &[usize], out: &[usize]) -> Vec<usize> {
    let rank = out.len();
    let padded = {
        let mut shape = vec![1usize; rank - src.len()];
        shape.extend_from_slice(src);
        shape
    };
    let mut strides = vec![1usize; rank];
    for position in (0..rank).rev() {
        strides[position] = if position + 1 < rank {
            strides[position + 1] * padded[position + 1]
        } else {
            1
        };
    }
    let mut out_strides = vec![1usize; rank];
    for position in (0..rank).rev() {
        out_strides[position] = if position + 1 < rank {
            out_strides[position + 1] * out[position + 1]
        } else {
            1
        };
    }
    let total: usize = out.iter().product();
    (0..total)
        .map(|flat| {
            let mut offset = 0;
            let mut remainder = flat;
            for position in 0..rank {
                let stride = out_strides[position];
                let index = remainder / stride;
                remainder %= stride;
                let source_index = if padded[position] == 1 { 0 } else { index };
                offset += source_index * strides[position];
            }
            offset
        })
        .collect()
}

/// Matrix and matrix-vector multiplication.
pub fn matmul(lhs: &Value, rhs: &Value) -> Result<Value> {
    let (Value::Tensor(left), Value::Tensor(right)) = (lhs, rhs) else {
        return Err(RuntimeError::Type("matmul expects two tensors".into()));
    };
    match (left.shape.len(), right.shape.len()) {
        (2, 2) => matmul_matrix(left, right),
        (2, 1) => matmul_matrix_vector(left, right),
        (1, 2) => matmul_vector_matrix(left, right),
        (1, 1) => matmul_vector_vector(left, right),
        _ => Err(RuntimeError::Arithmetic(
            "`@` expects a matrix and a vector or matrix".into(),
        )),
    }
}

#[cfg(not(feature = "faer"))]
fn matmul_matrix(left: &Tensor, right: &Tensor) -> Result<Value> {
    let (m, k) = (left.shape[0], left.shape[1]);
    let (k2, n) = (right.shape[0], right.shape[1]);
    if k != k2 {
        return Err(RuntimeError::Arithmetic(format!(
            "inner dimensions disagree: {k} vs {k2}"
        )));
    }
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += left.data[i * k + p] * right.data[p * n + j];
            }
            data[i * n + j] = acc;
        }
    }
    Ok(Value::Tensor(Tensor::new(
        unify_elem(left.elem, right.elem),
        vec![m, n],
        data,
    )))
}

/// Matrix multiplication backed by `faer` when the feature is enabled.
#[cfg(feature = "faer")]
fn matmul_matrix(left: &Tensor, right: &Tensor) -> Result<Value> {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat, Par};

    let (m, k) = (left.shape[0], left.shape[1]);
    let (k2, n) = (right.shape[0], right.shape[1]);
    if k != k2 {
        return Err(RuntimeError::Arithmetic(format!(
            "inner dimensions disagree: {k} vs {k2}"
        )));
    }
    let lhs = Mat::from_fn(m, k, |i, j| left.data[i * k + j]);
    let rhs = Mat::from_fn(k, n, |i, j| right.data[i * n + j]);
    let mut product = Mat::<f64>::zeros(m, n);
    matmul(&mut product, Accum::Replace, &lhs, &rhs, 1.0, Par::Seq);
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            data.push(product[(i, j)]);
        }
    }
    Ok(Value::Tensor(Tensor::new(
        unify_elem(left.elem, right.elem),
        vec![m, n],
        data,
    )))
}

fn matmul_matrix_vector(left: &Tensor, right: &Tensor) -> Result<Value> {
    let (m, k) = (left.shape[0], left.shape[1]);
    let k2 = right.shape[0];
    if k != k2 {
        return Err(RuntimeError::Arithmetic(format!(
            "inner dimensions disagree: {k} vs {k2}"
        )));
    }
    let data = (0..m)
        .map(|i| (0..k).map(|p| left.data[i * k + p] * right.data[p]).sum())
        .collect();
    Ok(Value::Tensor(Tensor::new(
        unify_elem(left.elem, right.elem),
        vec![m],
        data,
    )))
}

fn matmul_vector_matrix(left: &Tensor, right: &Tensor) -> Result<Value> {
    let k = left.shape[0];
    let (k2, n) = (right.shape[0], right.shape[1]);
    if k != k2 {
        return Err(RuntimeError::Arithmetic(format!(
            "inner dimensions disagree: {k} vs {k2}"
        )));
    }
    let data = (0..n)
        .map(|j| (0..k).map(|p| left.data[p] * right.data[p * n + j]).sum())
        .collect();
    Ok(Value::Tensor(Tensor::new(
        unify_elem(left.elem, right.elem),
        vec![n],
        data,
    )))
}

fn matmul_vector_vector(left: &Tensor, right: &Tensor) -> Result<Value> {
    let k = left.shape[0];
    let k2 = right.shape[0];
    if k != k2 {
        return Err(RuntimeError::Arithmetic(format!(
            "inner dimensions disagree: {k} vs {k2}"
        )));
    }
    let elem = unify_elem(left.elem, right.elem);
    let dot = (0..k).map(|p| left.data[p] * right.data[p]).sum();
    Ok(Value::Tensor(Tensor::scalar(elem, dot)))
}

/// Builds a tensor from runtime elements.
pub fn make_tensor(elements: &[Value]) -> Result<Value> {
    if elements.is_empty() {
        return Ok(Value::Tensor(Tensor::new(
            ElemType::F64,
            vec![0],
            Vec::new(),
        )));
    }
    if elements.iter().all(|value| !value.is_tensor()) {
        let data = elements
            .iter()
            .map(|value| {
                value.as_f64().ok_or_else(|| {
                    RuntimeError::Type(format!(
                        "tensor element {} is not numeric",
                        value.type_name()
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let elem = if elements
            .iter()
            .any(|value| matches!(value, Value::Float(_)))
        {
            ElemType::F64
        } else {
            ElemType::I64
        };
        return Ok(Value::Tensor(Tensor::new(elem, vec![elements.len()], data)));
    }

    let inner_shapes: Vec<&[usize]> = elements.iter().filter_map(Value::shape).collect();
    if inner_shapes.len() != elements.len() {
        return Err(RuntimeError::Type(
            "cannot mix scalars and tensors in one literal".into(),
        ));
    }
    let first = inner_shapes[0];
    if !inner_shapes.iter().all(|shape| *shape == first) {
        return Err(RuntimeError::Arithmetic("ragged tensor literal".into()));
    }
    let mut shape = vec![elements.len()];
    shape.extend_from_slice(first);
    let mut data = Vec::new();
    let mut elem = ElemType::I64;
    for value in elements {
        if let Value::Tensor(tensor) = value {
            if matches!(tensor.elem, ElemType::F64) {
                elem = ElemType::F64;
            }
            data.extend_from_slice(&tensor.data);
        }
    }
    Ok(Value::Tensor(Tensor::new(elem, shape, data)))
}

/// Indexes along axis 0 with a runtime index.
pub fn index(target: &Value, index: &Value) -> Result<Value> {
    let Value::Tensor(tensor) = target else {
        return Err(RuntimeError::Type("can only index a tensor".into()));
    };
    let position = index
        .as_i64()
        .ok_or_else(|| RuntimeError::Type("index must be an integer".into()))?;
    extract_tensor(tensor, position)
}

/// Extracts the `position`-th slice along axis 0.
pub fn extract(tensor: &Tensor, position: i64) -> Result<Value> {
    extract_tensor(tensor, position)
}

fn extract_tensor(tensor: &Tensor, position: i64) -> Result<Value> {
    if tensor.shape.is_empty() {
        return Err(RuntimeError::Type("cannot index a scalar".into()));
    }
    let axis = tensor.shape[0];
    if position < 0 || position as usize >= axis {
        return Err(RuntimeError::Arithmetic(format!(
            "index {position} out of bounds for axis of length {axis}"
        )));
    }
    let position = position as usize;
    if tensor.shape.len() == 1 {
        return Ok(number(tensor.elem, tensor.data[position]));
    }
    let row = tensor.data.len() / axis;
    let start = position * row;
    let shape = tensor.shape[1..].to_vec();
    Ok(Value::Tensor(Tensor::new(
        tensor.elem,
        shape,
        tensor.data[start..start + row].to_vec(),
    )))
}

/// Static extent of an axis.
pub fn dimension(tensor: &Tensor, axis: u32) -> Result<Value> {
    let axis = axis as usize;
    tensor
        .shape
        .get(axis)
        .map(|extent| Value::Int(*extent as i64))
        .ok_or_else(|| RuntimeError::Arithmetic(format!("axis {axis} out of range")))
}

fn number(elem: ElemType, value: f64) -> Value {
    match elem {
        ElemType::I64 => Value::Int(value as i64),
        ElemType::F32 | ElemType::F64 => Value::Float(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_arithmetic_stays_integral() {
        assert_eq!(
            binary(BinaryOp::Add, &Value::Int(2), &Value::Int(3)).unwrap(),
            Value::Int(5)
        );
        assert_eq!(
            binary(BinaryOp::Mul, &Value::Int(2), &Value::Float(3.0)).unwrap(),
            Value::Float(6.0)
        );
    }

    #[test]
    fn comparisons_yield_bools() {
        assert_eq!(
            binary(BinaryOp::Lt, &Value::Float(1.0), &Value::Float(2.0)).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            binary(BinaryOp::Eq, &Value::Int(1), &Value::Int(2)).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn division_by_zero_is_an_error() {
        assert!(binary(BinaryOp::Div, &Value::Int(1), &Value::Int(0)).is_err());
    }

    #[test]
    fn make_tensor_stacks_rows() {
        let row_a = Value::Tensor(Tensor::new(ElemType::F64, vec![2], vec![1.0, 2.0]));
        let row_b = Value::Tensor(Tensor::new(ElemType::F64, vec![2], vec![3.0, 4.0]));
        let stacked = make_tensor(&[row_a, row_b]).unwrap();
        assert_eq!(stacked.shape(), Some(&[2, 2][..]));
    }

    #[test]
    fn matmul_multiplies_matrices() {
        let a = Value::Tensor(Tensor::new(ElemType::F64, vec![1, 2], vec![1.0, 2.0]));
        let b = Value::Tensor(Tensor::new(ElemType::F64, vec![2, 1], vec![3.0, 4.0]));
        let product = matmul(&a, &b).unwrap();
        assert_eq!(
            product,
            Value::Tensor(Tensor::new(ElemType::F64, vec![1, 1], vec![11.0]))
        );
    }

    #[test]
    fn matmul_matrix_vector() {
        let matrix = Value::Tensor(Tensor::new(
            ElemType::F64,
            vec![2, 2],
            vec![1.0, 2.0, 3.0, 4.0],
        ));
        let vector = Value::Tensor(Tensor::new(ElemType::F64, vec![2], vec![1.0, 1.0]));
        let product = matmul(&matrix, &vector).unwrap();
        assert_eq!(
            product,
            Value::Tensor(Tensor::new(ElemType::F64, vec![2], vec![3.0, 7.0]))
        );
    }

    #[test]
    fn extract_row_of_matrix() {
        let matrix = Tensor::new(ElemType::F64, vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let row = extract(&matrix, 1).unwrap();
        assert_eq!(
            row,
            Value::Tensor(Tensor::new(ElemType::F64, vec![2], vec![3.0, 4.0]))
        );
    }

    fn tensor(shape: Vec<usize>, data: Vec<f64>) -> Value {
        Value::Tensor(Tensor::new(ElemType::F64, shape, data))
    }

    #[test]
    fn tensor_scalar_arithmetic_broadcasts() {
        let values = tensor(vec![3], vec![1.0, 2.0, 3.0]);
        assert_eq!(
            binary(BinaryOp::Add, &values, &Value::Int(10)).unwrap(),
            tensor(vec![3], vec![11.0, 12.0, 13.0])
        );
        assert_eq!(
            binary(BinaryOp::Mul, &Value::Float(2.0), &values).unwrap(),
            tensor(vec![3], vec![2.0, 4.0, 6.0])
        );
    }

    #[test]
    fn tensor_scalar_comparison_reduces_to_bool() {
        let values = tensor(vec![3], vec![1.0, 2.0, 3.0]);
        assert_eq!(
            binary(BinaryOp::Lt, &values, &Value::Int(4)).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            binary(BinaryOp::Lt, &values, &Value::Int(3)).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn tensor_broadcast_rank_extension() {
        let matrix = tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let column = tensor(vec![3], vec![10.0, 20.0, 30.0]);
        assert_eq!(
            binary(BinaryOp::Add, &matrix, &column).unwrap(),
            tensor(vec![2, 3], vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0])
        );
    }

    #[test]
    fn tensor_broadcast_with_singleton_axis() {
        let column = tensor(vec![2, 1], vec![1.0, 2.0]);
        let row = tensor(vec![1, 3], vec![10.0, 20.0, 30.0]);
        assert_eq!(
            binary(BinaryOp::Mul, &column, &row).unwrap(),
            tensor(vec![2, 3], vec![10.0, 20.0, 30.0, 20.0, 40.0, 60.0])
        );
    }

    #[test]
    fn incompatible_shapes_are_rejected() {
        let a = tensor(vec![2, 3], vec![0.0; 6]);
        let b = tensor(vec![2], vec![0.0; 2]);
        assert!(binary(BinaryOp::Add, &a, &b).is_err());
    }

    #[test]
    fn matmul_of_vectors_is_a_scalar() {
        let a = tensor(vec![3], vec![1.0, 2.0, 3.0]);
        let b = tensor(vec![3], vec![4.0, 5.0, 6.0]);
        let product = matmul(&a, &b).unwrap();
        assert_eq!(tensor(vec![], vec![32.0]), product);
    }

    #[cfg(feature = "faer")]
    #[test]
    fn faer_matmul_matches_the_reference_product() {
        let a = tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = tensor(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let product = matmul(&a, &b).unwrap();
        assert_eq!(product, tensor(vec![2, 2], vec![58.0, 64.0, 139.0, 154.0]));
    }
}

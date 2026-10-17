//! Implementations of Apeiron's built-in functions.
//!
//! Every entry point takes a slice of [`Value`]s, checks its own arity first
//! (so a wrong-argument-count call is reported as such rather than as a type
//! error), and returns a [`Value`] or a [`RuntimeError`].

use apeiron_ir::ElemType;
use apeiron_runtime::{Result, RuntimeError, Tensor, Value};

/// Rejects a call whose argument count does not match the signature.
fn expect(args: &[Value], count: usize, name: &str) -> Result<()> {
    if args.len() == count {
        Ok(())
    } else {
        Err(RuntimeError::Type(format!(
            "`{name}` expects {count} argument(s), found {}",
            args.len()
        )))
    }
}

/// Borrows the tensor at `index`.
fn arg_tensor<'a>(args: &'a [Value], index: usize, name: &str) -> Result<&'a Tensor> {
    match args.get(index) {
        Some(Value::Tensor(tensor)) => Ok(tensor),
        Some(other) => Err(RuntimeError::Type(format!(
            "argument {} of `{name}`: expected tensor, found {}",
            index + 1,
            other.type_name()
        ))),
        None => Err(RuntimeError::Type(format!(
            "`{name}` is missing argument {}",
            index + 1
        ))),
    }
}

/// Reads the integer at `index`.
fn arg_int(args: &[Value], index: usize, name: &str) -> Result<i64> {
    match args.get(index) {
        Some(Value::Int(value)) => Ok(*value),
        Some(other) => Err(RuntimeError::Type(format!(
            "argument {} of `{name}`: expected i64, found {}",
            index + 1,
            other.type_name()
        ))),
        None => Err(RuntimeError::Type(format!(
            "`{name}` is missing argument {}",
            index + 1
        ))),
    }
}

/// Applies `f` element-wise, preserving element type and shape.
fn map(tensor: &Tensor, f: impl Fn(f64) -> f64) -> Value {
    Value::Tensor(Tensor::new(
        tensor.elem,
        tensor.shape.clone(),
        tensor.data.iter().map(|value| f(*value)).collect(),
    ))
}

/// Applies `f` element-wise, widening the result to `f64`.
///
/// Used by transcendental functions whose outputs are generally fractional,
/// so an integer-tagged input does not silently truncate on later indexing.
fn map_float(tensor: &Tensor, f: impl Fn(f64) -> f64) -> Value {
    Value::Tensor(Tensor::new(
        ElemType::F64,
        tensor.shape.clone(),
        tensor.data.iter().map(|value| f(*value)).collect(),
    ))
}

/// Length guard for constructors built from a runtime count.
fn checked_length(count: i64, name: &str) -> Result<usize> {
    if count < 0 {
        return Err(RuntimeError::Type(format!(
            "`{name}` cannot build a tensor of length {count}"
        )));
    }
    const MAX_LENGTH: i64 = 1 << 24;
    if count > MAX_LENGTH {
        return Err(RuntimeError::Arithmetic(format!(
            "`{name}` refuses to allocate {count} elements (limit {MAX_LENGTH})"
        )));
    }
    Ok(count as usize)
}

// ── similarity and norms ─────────────────────────────────────────────

/// Cosine of the angle between two flattened tensors.
pub(crate) fn cosine_similarity(args: &[Value]) -> Result<Value> {
    expect(args, 2, "cosine_similarity")?;
    let left = arg_tensor(args, 0, "cosine_similarity")?;
    let right = arg_tensor(args, 1, "cosine_similarity")?;
    if left.data.len() != right.data.len() {
        return Err(RuntimeError::Arithmetic(format!(
            "cosine_similarity expects equal-length tensors, found {} and {}",
            left.data.len(),
            right.data.len()
        )));
    }
    let dot: f64 = left.data.iter().zip(&right.data).map(|(a, b)| a * b).sum();
    let norm_left: f64 = left.data.iter().map(|value| value * value).sum();
    let norm_right: f64 = right.data.iter().map(|value| value * value).sum();
    if norm_left == 0.0 || norm_right == 0.0 {
        return Err(RuntimeError::Arithmetic(
            "cosine_similarity is undefined for a zero vector".into(),
        ));
    }
    Ok(Value::Float(dot / norm_left.sqrt() / norm_right.sqrt()))
}

/// Inner product of two flattened tensors.
pub(crate) fn dot(args: &[Value]) -> Result<Value> {
    expect(args, 2, "dot")?;
    let left = arg_tensor(args, 0, "dot")?;
    let right = arg_tensor(args, 1, "dot")?;
    if left.data.len() != right.data.len() {
        return Err(RuntimeError::Arithmetic(format!(
            "dot expects equal-length tensors, found {} and {}",
            left.data.len(),
            right.data.len()
        )));
    }
    let total: f64 = left.data.iter().zip(&right.data).map(|(a, b)| a * b).sum();
    Ok(Value::Float(total))
}

/// Euclidean length of a flattened tensor.
pub(crate) fn norm(args: &[Value]) -> Result<Value> {
    expect(args, 1, "norm")?;
    let tensor = arg_tensor(args, 0, "norm")?;
    let total: f64 = tensor.data.iter().map(|value| value * value).sum();
    Ok(Value::Float(total.sqrt()))
}

// ── element-wise activations ─────────────────────────────────────────

pub(crate) fn relu(args: &[Value]) -> Result<Value> {
    expect(args, 1, "relu")?;
    let tensor = arg_tensor(args, 0, "relu")?;
    Ok(map(tensor, |value| value.max(0.0)))
}

pub(crate) fn sigmoid(args: &[Value]) -> Result<Value> {
    expect(args, 1, "sigmoid")?;
    let tensor = arg_tensor(args, 0, "sigmoid")?;
    Ok(map_float(tensor, |value| 1.0 / (1.0 + (-value).exp())))
}

pub(crate) fn tanh(args: &[Value]) -> Result<Value> {
    expect(args, 1, "tanh")?;
    let tensor = arg_tensor(args, 0, "tanh")?;
    Ok(map_float(tensor, f64::tanh))
}

pub(crate) fn exp(args: &[Value]) -> Result<Value> {
    expect(args, 1, "exp")?;
    let tensor = arg_tensor(args, 0, "exp")?;
    Ok(map_float(tensor, f64::exp))
}

pub(crate) fn log(args: &[Value]) -> Result<Value> {
    expect(args, 1, "log")?;
    let tensor = arg_tensor(args, 0, "log")?;
    if let Some(value) = tensor.data.iter().find(|value| **value < 0.0) {
        return Err(RuntimeError::Arithmetic(format!(
            "log is undefined for the negative value {value}"
        )));
    }
    Ok(map_float(tensor, f64::ln))
}

pub(crate) fn sqrt(args: &[Value]) -> Result<Value> {
    expect(args, 1, "sqrt")?;
    let tensor = arg_tensor(args, 0, "sqrt")?;
    if let Some(value) = tensor.data.iter().find(|value| **value < 0.0) {
        return Err(RuntimeError::Arithmetic(format!(
            "sqrt is undefined for the negative value {value}"
        )));
    }
    Ok(map_float(tensor, f64::sqrt))
}

pub(crate) fn abs(args: &[Value]) -> Result<Value> {
    expect(args, 1, "abs")?;
    let tensor = arg_tensor(args, 0, "abs")?;
    Ok(map(tensor, f64::abs))
}

/// Softmax over the flattened tensor, shifted for numerical stability.
pub(crate) fn softmax(args: &[Value]) -> Result<Value> {
    expect(args, 1, "softmax")?;
    let tensor = arg_tensor(args, 0, "softmax")?;
    if tensor.data.is_empty() {
        return Err(RuntimeError::Type(
            "softmax expects a non-empty tensor".into(),
        ));
    }
    let largest = tensor
        .data
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let shifted: Vec<f64> = tensor
        .data
        .iter()
        .map(|value| (value - largest).exp())
        .collect();
    let total: f64 = shifted.iter().sum();
    if total == 0.0 {
        return Err(RuntimeError::Arithmetic(
            "softmax normalization underflowed to zero".into(),
        ));
    }
    Ok(Value::Tensor(Tensor::new(
        ElemType::F64,
        tensor.shape.clone(),
        shifted.iter().map(|value| value / total).collect(),
    )))
}

// ── reductions ───────────────────────────────────────────────────────

pub(crate) fn sum(args: &[Value]) -> Result<Value> {
    expect(args, 1, "sum")?;
    let tensor = arg_tensor(args, 0, "sum")?;
    Ok(Value::Float(tensor.data.iter().sum()))
}

pub(crate) fn mean(args: &[Value]) -> Result<Value> {
    expect(args, 1, "mean")?;
    let tensor = arg_tensor(args, 0, "mean")?;
    if tensor.data.is_empty() {
        return Err(RuntimeError::Arithmetic(
            "mean of an empty tensor is undefined".into(),
        ));
    }
    Ok(Value::Float(
        tensor.data.iter().sum::<f64>() / tensor.data.len() as f64,
    ))
}

pub(crate) fn argmax(args: &[Value]) -> Result<Value> {
    expect(args, 1, "argmax")?;
    let tensor = arg_tensor(args, 0, "argmax")?;
    let position = tensor
        .data
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index);
    match position {
        Some(index) => Ok(Value::Int(index as i64)),
        None => Err(RuntimeError::Arithmetic(
            "argmax of an empty tensor is undefined".into(),
        )),
    }
}

/// Transposes a rank-2 tensor; ranks 0 and 1 are already their own transpose.
pub(crate) fn transpose(args: &[Value]) -> Result<Value> {
    expect(args, 1, "transpose")?;
    let tensor = arg_tensor(args, 0, "transpose")?;
    if tensor.rank() < 2 {
        return Ok(Value::Tensor(tensor.clone()));
    }
    if tensor.rank() > 2 {
        return Err(RuntimeError::Type(format!(
            "transpose expects a tensor of rank 0, 1, or 2, found rank {}",
            tensor.rank()
        )));
    }
    let (rows, columns) = (tensor.shape[0], tensor.shape[1]);
    if tensor.data.len() != rows * columns {
        return Err(RuntimeError::Arithmetic(
            "transpose: shape does not match the element count".into(),
        ));
    }
    let mut data = vec![0.0; tensor.data.len()];
    for row in 0..rows {
        for column in 0..columns {
            data[column * rows + row] = tensor.data[row * columns + column];
        }
    }
    Ok(Value::Tensor(Tensor::new(
        tensor.elem,
        vec![columns, rows],
        data,
    )))
}

// ── constructors ─────────────────────────────────────────────────────

pub(crate) fn zeros(args: &[Value]) -> Result<Value> {
    expect(args, 1, "zeros")?;
    let length = checked_length(arg_int(args, 0, "zeros")?, "zeros")?;
    Ok(Value::Tensor(Tensor::new(
        ElemType::F64,
        vec![length],
        vec![0.0; length],
    )))
}

pub(crate) fn ones(args: &[Value]) -> Result<Value> {
    expect(args, 1, "ones")?;
    let length = checked_length(arg_int(args, 0, "ones")?, "ones")?;
    Ok(Value::Tensor(Tensor::new(
        ElemType::F64,
        vec![length],
        vec![1.0; length],
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(
            ElemType::F64,
            vec![values.len()],
            values.to_vec(),
        ))
    }

    fn tensor_of(args: &[Value], index: usize) -> Tensor {
        match args.get(index) {
            Some(Value::Tensor(tensor)) => tensor.clone(),
            other => panic!("expected a tensor, found {other:?}"),
        }
    }

    fn float_of(result: Result<Value>) -> f64 {
        match result.unwrap() {
            Value::Float(value) => value,
            other => panic!("expected a float, found {other:?}"),
        }
    }

    #[test]
    fn cosine_similarity_of_identical_vectors_is_one() {
        let value = cosine_similarity(&[vector(&[1.0, 2.0, 3.0]), vector(&[1.0, 2.0, 3.0])]);
        assert!((float_of(value) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_similarity_of_orthogonal_vectors_is_zero() {
        let value = cosine_similarity(&[vector(&[1.0, 0.0]), vector(&[0.0, 1.0])]);
        assert!(float_of(value).abs() < 1e-12);
    }

    #[test]
    fn cosine_similarity_rejects_a_zero_vector() {
        let error = cosine_similarity(&[vector(&[0.0, 0.0]), vector(&[1.0, 1.0])]).unwrap_err();
        assert!(error.to_string().contains("zero vector"));
    }

    #[test]
    fn dot_and_norm_agree() {
        let a = vector(&[3.0, 4.0]);
        let b = vector(&[1.0, 0.0]);
        assert_eq!(float_of(dot(&[a.clone(), b.clone()])), 3.0);
        assert_eq!(float_of(norm(std::slice::from_ref(&a))), 5.0);
    }

    #[test]
    fn softmax_sums_to_one() {
        let tensor = softmax(&[vector(&[1.0, 2.0, 3.0])]).unwrap();
        let total: f64 = tensor_of(&[tensor], 0).data.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn activations_behave_as_expected() {
        let negatives = vector(&[-1.0, 0.0, 1.0]);
        let activated = relu(std::slice::from_ref(&negatives)).unwrap();
        assert_eq!(tensor_of(&[activated], 0).data, vec![0.0, 0.0, 1.0]);
        let midpoint = sigmoid(&[vector(&[0.0])]).unwrap();
        assert!((tensor_of(&[midpoint], 0).data[0] - 0.5).abs() < 1e-12);

        let stretched = tanh(&[vector(&[0.0])]).unwrap();
        assert_eq!(tensor_of(&[stretched], 0).data, vec![0.0]);
    }

    #[test]
    fn domain_helpers_reject_out_of_domain_inputs() {
        assert!(log(&[vector(&[-1.0])]).is_err());
        assert!(sqrt(&[vector(&[-1.0])]).is_err());
        assert!(log(&[vector(&[0.0])]).is_ok());
    }

    #[test]
    fn transcendental_ops_widen_integer_inputs_to_float() {
        let integers = Value::Tensor(Tensor::new(ElemType::I64, vec![2], vec![1.0, 4.0]));
        let rooted = tensor_of(&[sqrt(std::slice::from_ref(&integers)).unwrap()], 0);
        assert_eq!(rooted.elem, ElemType::F64);
        assert_eq!(rooted.data, vec![1.0, 2.0]);

        let squashed = tensor_of(&[sigmoid(std::slice::from_ref(&integers)).unwrap()], 0);
        assert_eq!(squashed.elem, ElemType::F64);
    }

    #[test]
    fn reductions_compute() {
        let values = vector(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(float_of(sum(std::slice::from_ref(&values))), 10.0);
        assert_eq!(float_of(mean(std::slice::from_ref(&values))), 2.5);
        assert_eq!(argmax(&[values]).unwrap(), Value::Int(3));
        assert!(mean(&[Value::Tensor(Tensor::new(ElemType::F64, vec![0], vec![]))]).is_err());
        assert!(argmax(&[Value::Tensor(Tensor::new(ElemType::F64, vec![0], vec![]))]).is_err());
    }

    #[test]
    fn transpose_swaps_a_matrix() {
        let matrix = Value::Tensor(Tensor::new(
            ElemType::F64,
            vec![2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ));
        let transposed = tensor_of(&[transpose(&[matrix]).unwrap()], 0);
        assert_eq!(transposed.shape, vec![3, 2]);
        assert_eq!(transposed.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn constructors_build_filled_vectors() {
        assert_eq!(
            tensor_of(&[zeros(&[Value::Int(3)]).unwrap()], 0).data,
            vec![0.0, 0.0, 0.0]
        );
        assert_eq!(
            tensor_of(&[ones(&[Value::Int(2)]).unwrap()], 0).data,
            vec![1.0, 1.0]
        );
        assert!(zeros(&[Value::Int(-1)]).is_err());
        assert!(zeros(&[Value::Float(3.0)]).is_err());
        assert!(ones(&[Value::Int(1 << 25)]).is_err());
    }

    #[test]
    fn wrong_arity_is_reported_before_types() {
        let error = sum(&[vector(&[1.0]), vector(&[2.0])]).unwrap_err();
        assert!(error
            .to_string()
            .contains("`sum` expects 1 argument(s), found 2"));

        let error = dot(&[vector(&[1.0])]).unwrap_err();
        assert!(error
            .to_string()
            .contains("`dot` expects 2 argument(s), found 1"));
    }
}

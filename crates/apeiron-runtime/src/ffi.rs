//! Loading of native externs declared in `Apeiron.toml`.
//!
//! An extern binds an Apeiron-level name to a symbol in a shared library. The
//! binding is described by [`ExternDecl`]; [`load_externs`] resolves each
//! symbol and returns a boxed callable that marshals [`Value`]s across the C
//! boundary. Externs support the primitive [`AbiType`] set and at most three
//! parameters.

use std::collections::HashMap;
use std::sync::Arc;

use apeiron_core::{AbiType, ExternDecl, ExternSig};

use crate::error::{Result, RuntimeError};
use crate::eval::NativeFn;
use crate::value::Value;

/// Generates one correctly-typed `extern "C"` trampoline for a signature.
macro_rules! mk {
    ($symbol:expr, $ret:ty, $ctor:path, [$( ($idx:literal, $pt:ty, $conv:ident, $name:ident) ),* $(,)?]) => {{
        let function: extern "C" fn($($pt),*) -> $ret = unsafe { std::mem::transmute($symbol) };
        Box::new(move |_args: &[Value]| {
            $(let $name = $conv(_args, $idx)?;)*
            Ok($ctor(function($($name),*)))
        }) as NativeFn
    }};
}

/// Loads every declared extern, returning callables keyed by extern name.
///
/// Each shared library is kept alive for as long as its externs are reachable,
/// so returned functions may be called safely at any later point.
pub fn load_externs(decls: &[ExternDecl]) -> Result<HashMap<String, NativeFn>> {
    let mut callables = HashMap::new();
    for decl in decls {
        let library = unsafe { libloading::Library::new(&decl.library) }.map_err(|error| {
            RuntimeError::Arithmetic(format!(
                "cannot load library `{}` for extern `{}`: {error}",
                decl.library, decl.name
            ))
        })?;
        let library = Arc::new(library);
        let symbol_name = decl.symbol().to_string();
        let symbol: *const () = unsafe {
            library
                .get::<*const ()>(symbol_name.as_bytes())
                .map(|symbol| *symbol)
        }
        .map_err(|error| {
            RuntimeError::Arithmetic(format!(
                "cannot find symbol `{symbol_name}` in `{}`: {error}",
                decl.library
            ))
        })?;

        let inner = build_callable(symbol, &decl.sig);
        // The library must outlive the symbol, so keep it alive alongside the
        // callable for as long as the extern is reachable.
        let handle = library;
        let callable: NativeFn = Box::new(move |args| {
            let _keep_loaded = &handle;
            inner(args)
        });
        callables.insert(decl.name.clone(), callable);
    }
    Ok(callables)
}

/// Builds a boxed callable for `symbol` under the calling convention `sig`.
fn build_callable(symbol: *const (), sig: &ExternSig) -> NativeFn {
    use AbiType::{F64, I64};
    match (sig.ret, sig.params.as_slice()) {
        (F64, []) => mk!(symbol, f64, Value::Float, []),
        (I64, []) => mk!(symbol, i64, Value::Int, []),

        (F64, [F64]) => mk!(symbol, f64, Value::Float, [(0, f64, arg_f64, a0)]),
        (F64, [I64]) => mk!(symbol, f64, Value::Float, [(0, i64, arg_i64, a0)]),
        (I64, [F64]) => mk!(symbol, i64, Value::Int, [(0, f64, arg_f64, a0)]),
        (I64, [I64]) => mk!(symbol, i64, Value::Int, [(0, i64, arg_i64, a0)]),

        (F64, [F64, F64]) => mk!(
            symbol,
            f64,
            Value::Float,
            [(0, f64, arg_f64, a0), (1, f64, arg_f64, a1)]
        ),
        (F64, [F64, I64]) => mk!(
            symbol,
            f64,
            Value::Float,
            [(0, f64, arg_f64, a0), (1, i64, arg_i64, a1)]
        ),
        (F64, [I64, F64]) => mk!(
            symbol,
            f64,
            Value::Float,
            [(0, i64, arg_i64, a0), (1, f64, arg_f64, a1)]
        ),
        (F64, [I64, I64]) => mk!(
            symbol,
            f64,
            Value::Float,
            [(0, i64, arg_i64, a0), (1, i64, arg_i64, a1)]
        ),
        (I64, [F64, F64]) => mk!(
            symbol,
            i64,
            Value::Int,
            [(0, f64, arg_f64, a0), (1, f64, arg_f64, a1)]
        ),
        (I64, [F64, I64]) => mk!(
            symbol,
            i64,
            Value::Int,
            [(0, f64, arg_f64, a0), (1, i64, arg_i64, a1)]
        ),
        (I64, [I64, F64]) => mk!(
            symbol,
            i64,
            Value::Int,
            [(0, i64, arg_i64, a0), (1, f64, arg_f64, a1)]
        ),
        (I64, [I64, I64]) => mk!(
            symbol,
            i64,
            Value::Int,
            [(0, i64, arg_i64, a0), (1, i64, arg_i64, a1)]
        ),

        (F64, [F64, F64, F64]) => mk!(
            symbol,
            f64,
            Value::Float,
            [
                (0, f64, arg_f64, a0),
                (1, f64, arg_f64, a1),
                (2, f64, arg_f64, a2)
            ]
        ),
        (F64, [F64, F64, I64]) => mk!(
            symbol,
            f64,
            Value::Float,
            [
                (0, f64, arg_f64, a0),
                (1, f64, arg_f64, a1),
                (2, i64, arg_i64, a2)
            ]
        ),
        (F64, [F64, I64, F64]) => mk!(
            symbol,
            f64,
            Value::Float,
            [
                (0, f64, arg_f64, a0),
                (1, i64, arg_i64, a1),
                (2, f64, arg_f64, a2)
            ]
        ),
        (F64, [F64, I64, I64]) => mk!(
            symbol,
            f64,
            Value::Float,
            [
                (0, f64, arg_f64, a0),
                (1, i64, arg_i64, a1),
                (2, i64, arg_i64, a2)
            ]
        ),
        (F64, [I64, F64, F64]) => mk!(
            symbol,
            f64,
            Value::Float,
            [
                (0, i64, arg_i64, a0),
                (1, f64, arg_f64, a1),
                (2, f64, arg_f64, a2)
            ]
        ),
        (F64, [I64, F64, I64]) => mk!(
            symbol,
            f64,
            Value::Float,
            [
                (0, i64, arg_i64, a0),
                (1, f64, arg_f64, a1),
                (2, i64, arg_i64, a2)
            ]
        ),
        (F64, [I64, I64, F64]) => mk!(
            symbol,
            f64,
            Value::Float,
            [
                (0, i64, arg_i64, a0),
                (1, i64, arg_i64, a1),
                (2, f64, arg_f64, a2)
            ]
        ),
        (F64, [I64, I64, I64]) => mk!(
            symbol,
            f64,
            Value::Float,
            [
                (0, i64, arg_i64, a0),
                (1, i64, arg_i64, a1),
                (2, i64, arg_i64, a2)
            ]
        ),
        (I64, [F64, F64, F64]) => mk!(
            symbol,
            i64,
            Value::Int,
            [
                (0, f64, arg_f64, a0),
                (1, f64, arg_f64, a1),
                (2, f64, arg_f64, a2)
            ]
        ),
        (I64, [F64, F64, I64]) => mk!(
            symbol,
            i64,
            Value::Int,
            [
                (0, f64, arg_f64, a0),
                (1, f64, arg_f64, a1),
                (2, i64, arg_i64, a2)
            ]
        ),
        (I64, [F64, I64, F64]) => mk!(
            symbol,
            i64,
            Value::Int,
            [
                (0, f64, arg_f64, a0),
                (1, i64, arg_i64, a1),
                (2, f64, arg_f64, a2)
            ]
        ),
        (I64, [F64, I64, I64]) => mk!(
            symbol,
            i64,
            Value::Int,
            [
                (0, f64, arg_f64, a0),
                (1, i64, arg_i64, a1),
                (2, i64, arg_i64, a2)
            ]
        ),
        (I64, [I64, F64, F64]) => mk!(
            symbol,
            i64,
            Value::Int,
            [
                (0, i64, arg_i64, a0),
                (1, f64, arg_f64, a1),
                (2, f64, arg_f64, a2)
            ]
        ),
        (I64, [I64, F64, I64]) => mk!(
            symbol,
            i64,
            Value::Int,
            [
                (0, i64, arg_i64, a0),
                (1, f64, arg_f64, a1),
                (2, i64, arg_i64, a2)
            ]
        ),
        (I64, [I64, I64, F64]) => mk!(
            symbol,
            i64,
            Value::Int,
            [
                (0, i64, arg_i64, a0),
                (1, i64, arg_i64, a1),
                (2, f64, arg_f64, a2)
            ]
        ),
        (I64, [I64, I64, I64]) => mk!(
            symbol,
            i64,
            Value::Int,
            [
                (0, i64, arg_i64, a0),
                (1, i64, arg_i64, a1),
                (2, i64, arg_i64, a2)
            ]
        ),

        // `ExternSig::parse` caps parameters at three, so anything else is a
        // malformed declaration that should fail if it is ever invoked.
        _ => Box::new(|_args: &[Value]| {
            Err(RuntimeError::Type("unsupported extern signature".into()))
        }) as NativeFn,
    }
}

/// Reads argument `index` as a C `double`.
fn arg_f64(args: &[Value], index: usize) -> Result<f64> {
    args.get(index)
        .and_then(Value::as_f64)
        .ok_or_else(|| RuntimeError::Type(format!("extern argument {index} must be a number")))
}

/// Reads argument `index` as a C `int64_t`.
fn arg_i64(args: &[Value], index: usize) -> Result<i64> {
    args.get(index)
        .and_then(Value::as_i64)
        .ok_or_else(|| RuntimeError::Type(format!("extern argument {index} must be an integer")))
}

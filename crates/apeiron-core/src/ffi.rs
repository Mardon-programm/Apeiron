//! Types describing foreign-function extern declarations.
//!
//! An extern binds an Apeiron-level name to a symbol in a native shared
//! library. The binding is described by a [`ExternDecl`]; its calling
//! convention is captured by an [`ExternSig`] such as `f64(f64, i64)`, which
//! only ever uses the primitive C types in [`AbiType`].

use std::fmt;

/// A primitive C ABI type an extern parameter or result may use.
///
/// Externs deliberately support a small, closed set of types so that the
/// calling convention can be established without guessing at layouts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AbiType {
    /// C `double`.
    F64,
    /// C `int64_t` (`long long`).
    I64,
}

impl AbiType {
    /// Parses an ABI type name such as `f64` or `i64`.
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim() {
            "f64" => Some(AbiType::F64),
            "i64" => Some(AbiType::I64),
            _ => None,
        }
    }
}

impl fmt::Display for AbiType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            AbiType::F64 => "f64",
            AbiType::I64 => "i64",
        })
    }
}

/// A parsed extern signature of the form `ret(arg, arg, ...)` such as
/// `f64(f64, f64)` or `i64()`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExternSig {
    /// The result type.
    pub ret: AbiType,
    /// The parameter types, in order.
    pub params: Vec<AbiType>,
}

impl ExternSig {
    /// Parses a signature such as `f64(f64, i64)`.
    ///
    /// Externs are capped at three parameters so the caller can establish the
    /// calling convention without combinatorial explosion.
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        let open = text
            .find('(')
            .ok_or_else(|| format!("missing `(` in signature `{text}`"))?;
        let close = text
            .rfind(')')
            .ok_or_else(|| format!("missing `)` in signature `{text}`"))?;
        if close < open {
            return Err(format!("misplaced `)` in signature `{text}`"));
        }
        let return_text = text[..open].trim();
        let ret = AbiType::parse(return_text)
            .ok_or_else(|| format!("unknown return type `{return_text}`"))?;

        let mut params = Vec::new();
        for chunk in text[open + 1..close].split(',') {
            let chunk = chunk.trim();
            if chunk.is_empty() {
                continue;
            }
            let parsed =
                AbiType::parse(chunk).ok_or_else(|| format!("unknown parameter type `{chunk}`"))?;
            params.push(parsed);
        }
        if params.len() > 3 {
            return Err(format!(
                "externs support at most 3 parameters, found {}",
                params.len()
            ));
        }
        Ok(ExternSig { ret, params })
    }
}

impl fmt::Display for ExternSig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.ret)?;
        for (index, param) in self.params.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{param}")?;
        }
        write!(f, ")")
    }
}

/// A declared external function: an Apeiron-level `name` bound to `symbol`
/// inside the shared library `library`, with calling convention `sig`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExternDecl {
    /// The Apeiron-level function name.
    pub name: String,
    /// Path to the shared library to load.
    pub library: String,
    /// The exported symbol; when empty, defaults to `name`.
    pub symbol: String,
    /// The calling convention.
    pub sig: ExternSig,
}

impl ExternDecl {
    /// The symbol to look up, falling back to the extern name.
    pub fn symbol(&self) -> &str {
        if self.symbol.is_empty() {
            &self.name
        } else {
            &self.symbol
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_two_argument_signature() {
        let sig = ExternSig::parse("f64(f64, i64)").unwrap();
        assert_eq!(sig.ret, AbiType::F64);
        assert_eq!(sig.params, vec![AbiType::F64, AbiType::I64]);
        assert_eq!(sig.to_string(), "f64(f64, i64)");
    }

    #[test]
    fn parses_a_nullary_signature() {
        let sig = ExternSig::parse("i64()").unwrap();
        assert_eq!(sig.ret, AbiType::I64);
        assert!(sig.params.is_empty());
    }

    #[test]
    fn unknown_types_are_rejected() {
        assert!(ExternSig::parse("void(f64)").is_err());
        assert!(ExternSig::parse("f64(str)").is_err());
        assert!(ExternSig::parse("f64").is_err());
    }

    #[test]
    fn too_many_parameters_are_rejected() {
        let error = ExternSig::parse("f64(f64, f64, f64, f64)").unwrap_err();
        assert!(error.contains("at most 3"));
    }

    #[test]
    fn symbol_falls_back_to_the_name() {
        let decl = ExternDecl {
            name: "cos".into(),
            library: "libm.so.6".into(),
            symbol: String::new(),
            sig: ExternSig::parse("f64(f64)").unwrap(),
        };
        assert_eq!(decl.symbol(), "cos");
    }
}

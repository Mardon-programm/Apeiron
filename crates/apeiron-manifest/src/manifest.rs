//! The project manifest format and its parser.
//!
//! A manifest is a deliberately small, TOML-flavoured text file called
//! `Apeiron.toml`. It supports top-level `key = value` pairs, `[section]`
//! headers, quoted or bare values, `#` comments, and blank lines:
//!
//! ```toml
//! name = "agent-lab"
//! version = "0.1.0"
//! entry = "main.ape"
//!
//! [modules]
//! planner = "src/planner.ape"
//!
//! [dependencies]
//! common = "../common"
//!
//! [native.cosine]
//! library = "libm.so.6"
//! symbol = "cos"
//! signature = "f64(f64)"
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use apeiron_core::{ExternDecl, ExternSig};

use crate::error::{Error, Result};

/// A parsed `Apeiron.toml`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    /// Project name.
    pub name: String,
    /// Project version; defaults to `0.1.0`.
    pub version: String,
    /// Entry-point file, relative to the manifest; defaults to `main.ape`.
    pub entry: String,
    /// Named source files declared under `[modules]`.
    pub modules: BTreeMap<String, String>,
    /// Path dependencies declared under `[dependencies]`, keyed by name.
    pub dependencies: BTreeMap<String, String>,
    /// Native externs declared under `[native.<name>]`, keyed by name.
    pub externs: BTreeMap<String, ExternDecl>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    Root,
    Modules,
    Dependencies,
    Native,
}

/// A partially-filled `[native.<name>]` block.
#[derive(Default)]
struct NativeBuilder {
    library: Option<(String, usize)>,
    symbol: Option<(String, usize)>,
    signature: Option<(String, usize)>,
}

/// Parses a manifest held in `source`, attributing errors to `path`.
pub fn parse_manifest(path: &Path, source: &str) -> Result<Manifest> {
    let mut name = None;
    let mut version = None;
    let mut entry = None;
    let mut modules = BTreeMap::new();
    let mut dependencies = BTreeMap::new();
    let mut natives: BTreeMap<String, NativeBuilder> = BTreeMap::new();
    let mut section = Section::Root;
    let mut native_name = String::new();

    for (index, raw) in source.lines().enumerate() {
        let line_number = index + 1;
        let line = strip_comment(raw);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(header) = line
            .strip_prefix('[')
            .and_then(|rest| rest.strip_suffix(']'))
        {
            let header = header.trim();
            if let Some(name) = header.strip_prefix("native.") {
                let name = name.trim();
                if name.is_empty() {
                    return Err(syntax(
                        path,
                        line_number,
                        "expected `[native.<name>]`".to_string(),
                    ));
                }
                if natives.contains_key(name) {
                    return Err(Error::Duplicate {
                        path: path.to_path_buf(),
                        key: format!("native.{name}"),
                    });
                }
                natives.insert(name.to_string(), NativeBuilder::default());
                section = Section::Native;
                native_name = name.to_string();
                continue;
            }
            section = match header {
                "" => Section::Root,
                "modules" => Section::Modules,
                "dependencies" => Section::Dependencies,
                "native" => {
                    return Err(syntax(
                        path,
                        line_number,
                        "expected `[native.<name>]`".to_string(),
                    ))
                }
                other => {
                    return Err(syntax(
                        path,
                        line_number,
                        format!("unknown section `[{other}]`"),
                    ))
                }
            };
            continue;
        }

        let Some((key, value)) = split_once_unquoted(line, '=') else {
            return Err(syntax(
                path,
                line_number,
                "expected `key = value`".to_string(),
            ));
        };
        let key = key.trim();
        if key.is_empty() {
            return Err(syntax(path, line_number, "empty key".to_string()));
        }
        let value =
            parse_value(value.trim()).map_err(|message| syntax(path, line_number, message))?;

        match section {
            Section::Root => match key {
                "name" => set_once(&mut name, value, path, key)?,
                "version" => set_once(&mut version, value, path, key)?,
                "entry" => set_once(&mut entry, value, path, key)?,
                other => return Err(syntax(path, line_number, format!("unknown key `{other}`"))),
            },
            Section::Modules => insert_unique(&mut modules, key, value, path)?,
            Section::Dependencies => insert_unique(&mut dependencies, key, value, path)?,
            Section::Native => {
                let builder = natives
                    .get_mut(&native_name)
                    .expect("native section always has a builder");
                let full_key = format!("native.{native_name}.{key}");
                match key {
                    "library" => {
                        set_once_line(&mut builder.library, value, line_number, &full_key, path)?
                    }
                    "symbol" => {
                        set_once_line(&mut builder.symbol, value, line_number, &full_key, path)?
                    }
                    "signature" => {
                        set_once_line(&mut builder.signature, value, line_number, &full_key, path)?
                    }
                    other => {
                        return Err(syntax(path, line_number, format!("unknown key `{other}`")))
                    }
                }
            }
        }
    }

    let name = name.ok_or_else(|| Error::MissingField {
        path: path.to_path_buf(),
        field: "name",
    })?;

    let externs = finalize_natives(path, natives)?;

    Ok(Manifest {
        name,
        version: version.unwrap_or_else(|| "0.1.0".to_string()),
        entry: entry.unwrap_or_else(|| "main.ape".to_string()),
        modules,
        dependencies,
        externs,
    })
}

/// Turns the accumulated `[native.<name>]` blocks into extern declarations.
fn finalize_natives(
    path: &Path,
    natives: BTreeMap<String, NativeBuilder>,
) -> Result<BTreeMap<String, ExternDecl>> {
    let mut externs = BTreeMap::new();
    for (name, builder) in natives {
        let Some((library, _)) = builder.library else {
            return Err(Error::MissingExternField {
                path: path.to_path_buf(),
                name: name.clone(),
                field: "library",
            });
        };
        let Some((signature, signature_line)) = builder.signature else {
            return Err(Error::MissingExternField {
                path: path.to_path_buf(),
                name: name.clone(),
                field: "signature",
            });
        };
        let sig = ExternSig::parse(&signature)
            .map_err(|message| syntax(path, signature_line, format!("native.{name}: {message}")))?;
        let symbol = builder.symbol.map(|(value, _)| value).unwrap_or_default();
        externs.insert(
            name.clone(),
            ExternDecl {
                name,
                library,
                symbol,
                sig,
            },
        );
    }
    Ok(externs)
}

fn syntax(path: &Path, line: usize, message: String) -> Error {
    Error::Syntax {
        path: path.to_path_buf(),
        line,
        message,
    }
}

fn set_once(slot: &mut Option<String>, value: String, path: &Path, key: &str) -> Result<()> {
    if slot.is_some() {
        return Err(Error::Duplicate {
            path: path.to_path_buf(),
            key: key.to_string(),
        });
    }
    *slot = Some(value);
    Ok(())
}

/// Sets a `(value, line)` slot once, reporting a duplicate with the full key.
fn set_once_line(
    slot: &mut Option<(String, usize)>,
    value: String,
    line: usize,
    full_key: &str,
    path: &Path,
) -> Result<()> {
    if slot.is_some() {
        return Err(Error::Duplicate {
            path: path.to_path_buf(),
            key: full_key.to_string(),
        });
    }
    *slot = Some((value, line));
    Ok(())
}

fn insert_unique(
    map: &mut BTreeMap<String, String>,
    key: &str,
    value: String,
    path: &Path,
) -> Result<()> {
    if map.contains_key(key) {
        return Err(Error::Duplicate {
            path: path.to_path_buf(),
            key: key.to_string(),
        });
    }
    map.insert(key.to_string(), value);
    Ok(())
}

/// Removes a trailing `# comment`, ignoring `#` inside a quoted string.
fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    let mut escaped = false;
    for (offset, character) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '#' if !in_string => return &line[..offset],
            _ => {}
        }
    }
    line
}

/// Splits on the first occurrence of `delimiter` outside a quoted string.
fn split_once_unquoted(line: &str, delimiter: char) -> Option<(&str, &str)> {
    let mut in_string = false;
    let mut escaped = false;
    for (offset, character) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            _ if character == delimiter && !in_string => {
                return Some((&line[..offset], &line[offset + character.len_utf8()..]));
            }
            _ => {}
        }
    }
    None
}

/// Parses a quoted string or a bare token.
fn parse_value(raw: &str) -> std::result::Result<String, String> {
    if let Some(inner) = raw.strip_prefix('"') {
        let Some(inner) = inner.strip_suffix('"') else {
            return Err("unterminated string value".to_string());
        };
        return Ok(unescape(inner));
    }
    if raw.is_empty() {
        return Err("missing value".to_string());
    }
    if raw.chars().any(char::is_whitespace) {
        return Err(format!("unquoted value `{raw}` contains whitespace"));
    }
    Ok(raw.to_string())
}

fn unescape(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(character) = chars.next() {
        if character == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(character);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn path() -> PathBuf {
        PathBuf::from("Apeiron.toml")
    }

    #[test]
    fn parses_a_full_manifest() {
        let source = r#"
# the project
name = "agent-lab"
version = "0.2.0"
entry = "app.ape"

[modules]
planner = "src/planner.ape"
executor = "src/executor.ape"

[dependencies]
common = "../common"
"#;
        let manifest = parse_manifest(&path(), source).unwrap();
        assert_eq!(manifest.name, "agent-lab");
        assert_eq!(manifest.version, "0.2.0");
        assert_eq!(manifest.entry, "app.ape");
        assert_eq!(manifest.modules["planner"], "src/planner.ape");
        assert_eq!(manifest.modules.len(), 2);
        assert_eq!(manifest.dependencies["common"], "../common");
    }

    #[test]
    fn defaults_fill_optional_fields() {
        let manifest = parse_manifest(&path(), "name = \"tiny\"\n").unwrap();
        assert_eq!(manifest.version, "0.1.0");
        assert_eq!(manifest.entry, "main.ape");
        assert!(manifest.modules.is_empty());
        assert!(manifest.externs.is_empty());
    }

    #[test]
    fn parses_a_native_extern() {
        let source = r#"
name = "ffi-demo"

[native.cosine]
library = "libm.so.6"
symbol = "cos"
signature = "f64(f64)"
"#;
        let manifest = parse_manifest(&path(), source).unwrap();
        let cosine = &manifest.externs["cosine"];
        assert_eq!(cosine.library, "libm.so.6");
        assert_eq!(cosine.symbol(), "cos");
        assert_eq!(cosine.sig.params.len(), 1);
    }

    #[test]
    fn native_symbol_defaults_to_the_extern_name() {
        let source = r#"
name = "ffi-demo"

[native.sqrt2]
library = "libm.so.6"
signature = "f64(f64)"
"#;
        let manifest = parse_manifest(&path(), source).unwrap();
        assert_eq!(manifest.externs["sqrt2"].symbol(), "sqrt2");
    }

    #[test]
    fn native_missing_library_is_reported() {
        let error = parse_manifest(&path(), "name = \"x\"\n[native.a]\nsignature = \"f64()\"\n")
            .unwrap_err();
        assert!(error.to_string().contains("library"));
    }

    #[test]
    fn native_bad_signature_is_reported() {
        let error = parse_manifest(
            &path(),
            "name = \"x\"\n[native.a]\nlibrary = \"l\"\nsignature = \"void()\"\n",
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown return type"));
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let source = "\n  # leading comment\nname = \"x\" # trailing\n\n";
        let manifest = parse_manifest(&path(), source).unwrap();
        assert_eq!(manifest.name, "x");
    }

    #[test]
    fn hashes_inside_strings_survive() {
        let manifest = parse_manifest(&path(), "name = \"a#b\"\n").unwrap();
        assert_eq!(manifest.name, "a#b");
    }

    #[test]
    fn escapes_are_expanded() {
        let manifest = parse_manifest(&path(), "name = \"line\\nbreak\"\n").unwrap();
        assert_eq!(manifest.name, "line\nbreak");
    }

    #[test]
    fn missing_name_is_reported() {
        let error = parse_manifest(&path(), "version = \"0.1.0\"\n").unwrap_err();
        assert!(matches!(error, Error::MissingField { field: "name", .. }));
    }

    #[test]
    fn unknown_section_is_reported() {
        let error = parse_manifest(&path(), "name = \"x\"\n[bogus]\n").unwrap_err();
        assert!(error.to_string().contains("unknown section"));
    }

    #[test]
    fn malformed_line_is_reported_with_a_line_number() {
        let error = parse_manifest(&path(), "name = \"x\"\nnot a pair\n").unwrap_err();
        match error {
            Error::Syntax { line, .. } => assert_eq!(line, 2),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn duplicate_keys_are_rejected() {
        let error = parse_manifest(&path(), "name = \"x\"\nname = \"y\"\n").unwrap_err();
        assert!(matches!(error, Error::Duplicate { .. }));
    }

    #[test]
    fn bare_values_are_accepted() {
        let manifest = parse_manifest(&path(), "name = agent\n").unwrap();
        assert_eq!(manifest.name, "agent");
    }
}

//! End-to-end tests for the CLI library and binary.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use apeiron_cli::{compile, format, parse, tokenize, Options};

const GOOD: &str = r#"
belief Threat {
    proposition: "anomaly",
    confidence: 0.40
}

belief Threat update {
    likelihood: 0.90,
    reliability: 0.90
}

cause Temperature -> Pressure
observe Temperature = 42.0

memory episodic ShortTermMem { dimension: 4, capacity: 100 }
remember ShortTermMem key = q value = "hello"
"#;

#[test]
fn compile_succeeds_and_runs() {
    let outcome = compile(
        GOOD,
        "good.ape",
        Options {
            strict: false,
            run: true,
        },
    );
    assert!(!outcome.has_errors(), "{}", outcome.render_diagnostics());
    assert!(outcome.ir.is_some());
    let state = outcome.state.expect("state");
    assert!(state.beliefs["Threat"].confidence > 0.40);
    assert_eq!(state.causal.edge_count(), 1);
    assert!(state.memories.contains_key("ShortTermMem"));
}

#[test]
fn syntax_errors_stop_the_pipeline() {
    let outcome = compile("tensor x = [1.0,", "bad.ape", Options::default());
    assert!(outcome.has_errors());
    assert!(outcome.ir.is_none());
    assert!(outcome.render_diagnostics().contains("PAR0001"));
}

#[test]
fn strict_mode_promotes_unresolved_to_error() {
    let source = "let y = unknown_identifier\n";
    let lenient = compile(source, "s.ape", Options::default());
    assert!(!lenient.has_errors());
    assert!(lenient.warning_count() >= 1);

    let strict = compile(
        source,
        "s.ape",
        Options {
            strict: true,
            run: false,
        },
    );
    assert!(strict.has_errors());
}

#[test]
fn state_summary_reports_sections() {
    let outcome = compile(
        GOOD,
        "good.ape",
        Options {
            strict: false,
            run: true,
        },
    );
    let summary = outcome.state_summary();
    assert!(summary.contains("beliefs"));
    assert!(summary.contains("memories"));
    assert!(summary.contains("causal"));
}

#[test]
fn tokenize_and_parse_produce_output() {
    let tokens = tokenize("tensor x = [1.0, 2.0]\n", "t.ape").expect("tokens");
    assert!(tokens.lines().count() > 1);

    let ast = parse("tensor x = [1.0, 2.0]\n", "t.ape").expect("ast");
    assert!(ast.contains("tensor x"));
}

#[test]
fn tokenize_reports_lex_errors() {
    let error = tokenize("let x = 'unterminated\n", "t.ape").unwrap_err();
    assert!(error.contains("LEX0001"));
}

fn write_temp(name: &str, contents: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("apeiron-cli-{}-{name}", std::process::id()));
    fs::write(&path, contents).expect("write temp");
    path
}

fn apeiron() -> Command {
    Command::new(env!("CARGO_BIN_EXE_apeiron"))
}

#[test]
fn binary_version_and_help() {
    let version = apeiron().arg("--version").output().expect("run");
    assert!(version.status.success());
    assert!(String::from_utf8_lossy(&version.stdout).contains("apeiron"));

    let help = apeiron().arg("--help").output().expect("run");
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("USAGE"));
}

#[test]
fn binary_run_reports_state() {
    let path = write_temp("run.ape", GOOD);
    let output = apeiron().arg("run").arg(&path).output().expect("run");
    let _ = fs::remove_file(&path);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("ran"));
    assert!(stdout.contains("beliefs"));
}

#[test]
fn binary_check_fails_on_bad_source() {
    let path = write_temp("bad.ape", "tensor x = [1.0,\n");
    let output = apeiron().arg("check").arg(&path).output().expect("run");
    let _ = fs::remove_file(&path);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("PAR0001"));
}

#[test]
fn binary_ir_prints_module() {
    let path = write_temp("ir.ape", "let x = 1 + 2\n");
    let output = apeiron().arg("ir").arg(&path).output().expect("run");
    let _ = fs::remove_file(&path);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("module apeiron"));
}

#[test]
fn binary_effects_prints_report() {
    let path = write_temp("effects.ape", "cause A -> B\n");
    let output = apeiron().arg("effects").arg(&path).output().expect("run");
    let _ = fs::remove_file(&path);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("module effects"));
    assert!(stdout.contains("causal"));
}

#[test]
fn binary_unknown_command_exits_with_two() {
    let output = apeiron().arg("frobnicate").output().expect("run");
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn binary_repl_evaluates_piped_input() {
    let mut child = apeiron()
        .arg("repl")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn repl");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(b"cause Length -> Threshold\nobserve Length = 5.0\n:state\n:quit\n")
        .expect("write input");
    let output = child.wait_with_output().expect("wait");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("causal"));
}

#[test]
fn binary_build_links_and_runs_a_project() {
    let dir = std::env::temp_dir().join(format!("apeiron-cli-build-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create project dir");
    fs::write(
        dir.join("Apeiron.toml"),
        "name = \"demo\"\nentry = \"main.ape\"\n",
    )
    .expect("write manifest");
    fs::write(
        dir.join("main.ape"),
        "cause Length -> Threshold\nobserve Length = 5.0\n",
    )
    .expect("write source");

    let output = apeiron().arg("build").arg(&dir).output().expect("run");
    let _ = fs::remove_dir_all(&dir);

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("built demo"));
    assert!(stdout.contains("causal"));
}

#[test]
fn format_normalises_and_is_idempotent() {
    let formatted = format("let x=1+2\n", "f.ape").expect("format");
    assert_eq!(formatted, "let x = 1 + 2\n");
    assert_eq!(format(&formatted, "f.ape").expect("format"), formatted);
}

#[test]
fn format_reports_lex_errors() {
    let error = format("/* nope", "f.ape").unwrap_err();
    assert!(error.contains("LEX0001"));
}

#[test]
fn binary_fmt_check_detects_and_fixes() {
    let path = write_temp("fmt.ape", "let x=1+2\n");

    let needs = apeiron()
        .arg("fmt")
        .arg("--check")
        .arg(&path)
        .output()
        .expect("run");
    assert!(!needs.status.success());

    let fixed = apeiron().arg("fmt").arg(&path).output().expect("run");
    assert!(fixed.status.success());
    assert_eq!(fs::read_to_string(&path).expect("read"), "let x = 1 + 2\n");

    let clean = apeiron()
        .arg("fmt")
        .arg("--check")
        .arg(&path)
        .output()
        .expect("run");
    let _ = fs::remove_file(&path);
    assert!(clean.status.success());
}

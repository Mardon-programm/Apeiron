//! End-to-end tests for project loading and linking.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use apeiron_manifest::{Error, Project, MANIFEST_FILE};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let mut root = std::env::temp_dir();
    root.push(format!(
        "apeiron-manifest-{}-{}-{}",
        std::process::id(),
        tag,
        COUNTER.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create temp dir");
    root
}

fn write_file(root: &Path, relative: &str, contents: &str) {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

fn project_dir(tag: &str, manifest: &str, files: &[(&str, &str)]) -> PathBuf {
    let root = temp_dir(tag);
    write_file(&root, MANIFEST_FILE, manifest);
    for (relative, contents) in files {
        write_file(&root, relative, contents);
    }
    root
}

#[test]
fn loads_and_links_a_single_file_project() {
    let root = project_dir(
        "single",
        "name = \"solo\"\nentry = \"main.ape\"\n",
        &[("main.ape", "let x = 1\n")],
    );
    let project = Project::load(&root).expect("load");
    assert_eq!(project.name(), "solo");
    assert!(project.link().unwrap().contains("let x = 1"));
    assert_eq!(project.source_files().len(), 1);
}

#[test]
fn modules_are_linked_before_the_entry() {
    let root = project_dir(
        "modules",
        "name = \"m\"\nentry = \"main.ape\"\n[modules]\nbase = \"base.ape\"\n",
        &[
            ("base.ape", "cause A -> B\n"),
            ("main.ape", "observe A = 1.0\n"),
        ],
    );
    let linked = Project::load(&root).expect("load").link().unwrap();
    let base = linked.find("cause A").unwrap();
    let main = linked.find("observe A").unwrap();
    assert!(base < main, "modules should precede the entry point");
}

#[test]
fn dependencies_are_linked_before_dependents() {
    let root = project_dir(
        "dep",
        "name = \"root\"\nentry = \"main.ape\"\n[dependencies]\ncommon = \"sub\"\n",
        &[("main.ape", "ROOT_MAIN\n")],
    );
    write_file(
        &root,
        "sub/Apeiron.toml",
        "name = \"common\"\nentry = \"lib.ape\"\n",
    );
    write_file(&root, "sub/lib.ape", "DEP_LIB\n");

    let project = Project::load(&root).expect("load");
    assert_eq!(project.dependencies.len(), 1);
    assert_eq!(project.dependencies[0].name(), "common");

    let linked = project.link().unwrap();
    let dep = linked.find("DEP_LIB").unwrap();
    let main = linked.find("ROOT_MAIN").unwrap();
    assert!(dep < main, "dependencies should precede the dependent");
}

#[test]
fn missing_manifest_is_reported() {
    let root = temp_dir("missing");
    let error = Project::load(&root).unwrap_err();
    match error {
        Error::NotFound { path } => assert!(path.ends_with(MANIFEST_FILE)),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn dependency_cycles_are_detected() {
    let base = temp_dir("cycle");
    write_file(
        &base,
        "a/Apeiron.toml",
        "name = \"a\"\nentry = \"main.ape\"\n[dependencies]\nb = \"../b\"\n",
    );
    write_file(&base, "a/main.ape", "");
    write_file(
        &base,
        "b/Apeiron.toml",
        "name = \"b\"\nentry = \"main.ape\"\n[dependencies]\na = \"../a\"\n",
    );
    write_file(&base, "b/main.ape", "");

    let error = Project::load(base.join("a")).unwrap_err();
    assert!(matches!(error, Error::Cycle { .. }));
}

#[test]
fn shared_dependency_is_not_a_cycle() {
    let base = temp_dir("diamond");
    write_file(
        &base,
        "app/Apeiron.toml",
        "name = \"app\"\nentry = \"main.ape\"\n[dependencies]\nleft = \"../left\"\nright = \"../right\"\n",
    );
    write_file(&base, "app/main.ape", "");
    for side in ["left", "right"] {
        write_file(
            &base,
            &format!("{side}/Apeiron.toml"),
            "name = \"side\"\nentry = \"main.ape\"\n[dependencies]\ncommon = \"../common\"\n",
        );
        write_file(&base, &format!("{side}/main.ape"), "");
    }
    write_file(
        &base,
        "common/Apeiron.toml",
        "name = \"common\"\nentry = \"main.ape\"\n",
    );
    write_file(&base, "common/main.ape", "");

    let project = Project::load(base.join("app")).expect("shared deps are fine");
    assert_eq!(project.dependencies.len(), 2);
}

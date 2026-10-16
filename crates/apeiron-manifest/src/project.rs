//! Projects: a manifest plus its resolved dependencies and source files.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::manifest::{parse_manifest, Manifest};

/// The file a project's manifest lives in.
pub const MANIFEST_FILE: &str = "Apeiron.toml";

/// A loaded project: the root manifest together with its path dependencies.
#[derive(Debug)]
pub struct Project {
    /// Directory containing the manifest.
    pub root: PathBuf,
    /// The parsed root manifest.
    pub manifest: Manifest,
    /// Path dependencies, in declaration order.
    pub dependencies: Vec<Project>,
}

impl Project {
    /// Loads a project rooted at `root`, resolving dependencies recursively.
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let mut stack = Vec::new();
        Self::load_with_stack(root.as_ref().to_path_buf(), &mut stack)
    }

    /// The project's name.
    pub fn name(&self) -> &str {
        &self.manifest.name
    }

    /// Walks the dependency graph and returns every source file to compile,
    /// dependencies first, then this project's modules, then its entry.
    ///
    /// Files are returned in the order they should be linked.
    pub fn source_files(&self) -> Vec<PathBuf> {
        let mut files = Vec::new();
        self.collect_source_files(&mut files);
        files
    }

    /// Reads every source file and concatenates them into one program.
    ///
    /// The result is ready to hand to the compiler pipeline as a single module.
    pub fn link(&self) -> Result<String> {
        let mut combined = String::new();
        for file in self.source_files() {
            let source = read(&file)?;
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str(&source);
        }
        Ok(combined)
    }

    fn collect_source_files(&self, out: &mut Vec<PathBuf>) {
        for dependency in &self.dependencies {
            dependency.collect_source_files(out);
        }
        for relative in self.manifest.modules.values() {
            out.push(self.root.join(relative));
        }
        out.push(self.root.join(&self.manifest.entry));
    }

    fn load_with_stack(root: PathBuf, stack: &mut Vec<PathBuf>) -> Result<Self> {
        let canonical = fs::canonicalize(&root).unwrap_or_else(|_| root.clone());
        if stack.contains(&canonical) {
            return Err(Error::Cycle { path: root });
        }

        let manifest_path = root.join(MANIFEST_FILE);
        let source = read_manifest(&manifest_path)?;
        let manifest = parse_manifest(&manifest_path, &source)?;

        stack.push(canonical);
        let mut dependencies = Vec::new();
        for relative in manifest.dependencies.values() {
            let dependency_root = root.join(relative);
            match Self::load_with_stack(dependency_root, stack) {
                Ok(project) => dependencies.push(project),
                Err(error) => {
                    stack.pop();
                    return Err(error);
                }
            }
        }
        stack.pop();

        Ok(Self {
            root,
            manifest,
            dependencies,
        })
    }
}

fn read(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_manifest(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|source| {
        if source.kind() == io::ErrorKind::NotFound {
            Error::NotFound {
                path: path.to_path_buf(),
            }
        } else {
            Error::Io {
                path: path.to_path_buf(),
                source,
            }
        }
    })
}

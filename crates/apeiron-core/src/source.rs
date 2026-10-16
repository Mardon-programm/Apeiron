//! Source file registry used to resolve byte offsets into `line:column`
//! coordinates and to slice spans back into displayable source text.

use std::fmt;
use std::ops::Range;

use crate::error::CoreError;
use crate::span::{Location, Span};

/// Handle to a file registered in a [`SourceMap`].
///
/// Copyable and cheap to compare; becomes meaningful only relative to
/// the map that produced it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct FileId(pub u32);

impl fmt::Display for FileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One registered source file together with its precomputed line index.
#[derive(Clone, Debug)]
pub struct SourceFile {
    id: FileId,
    name: String,
    source: String,
    line_starts: Vec<usize>,
}

impl SourceFile {
    fn new(id: FileId, name: String, source: String) -> Self {
        let line_starts = compute_line_starts(&source);
        Self {
            id,
            name,
            source,
            line_starts,
        }
    }

    /// Handle of this file inside its map.
    pub fn id(&self) -> FileId {
        self.id
    }

    /// Display name (usually a path such as `agent.ape`).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Full file contents.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Number of lines (always at least 1).
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// Byte range of the 1-based `line`, excluding its line terminator.
    ///
    /// Returns `None` when the line number is out of range.
    pub fn line_range(&self, line: u32) -> Option<Range<usize>> {
        let idx = line.checked_sub(1)? as usize;
        let start = *self.line_starts.get(idx)?;
        let end = match self.line_starts.get(idx + 1) {
            Some(next) => trim_line_terminator(&self.source, start..*next),
            None => self.source.len(),
        };
        Some(start..end)
    }

    /// Text of the 1-based `line` without its terminator.
    pub fn line_text(&self, line: u32) -> Option<&str> {
        let range = self.line_range(line)?;
        self.source.get(range)
    }

    /// Converts a byte offset into a 1-based `line`/`column` pair.
    ///
    /// Out-of-range offsets are clamped to the file boundaries and
    /// offsets landing inside a multi-byte character are snapped down to
    /// the enclosing character boundary, so diagnostics generated near
    /// EOF or in UTF-8 text still point somewhere sensible.
    pub fn line_col(&self, offset: usize) -> (u32, u32) {
        let mut offset = offset.min(self.source.len());
        while offset > 0 && !self.source.is_char_boundary(offset) {
            offset -= 1;
        }
        let line_idx = self.line_index(offset);
        let line_start = self.line_starts[line_idx];
        let column = self.source[line_start..offset].chars().count() as u32 + 1;
        (line_idx as u32 + 1, column)
    }

    /// Builds a [`Location`] for a byte offset inside this file.
    pub fn location(&self, offset: usize) -> Location {
        let offset = offset.min(self.source.len());
        let (line, column) = self.line_col(offset);
        Location::new(self.id, offset as u32, line, column)
    }

    /// Resolves the text covered by `span` in this file.
    pub fn span_text(&self, span: Span) -> Option<&str> {
        if span.file() != self.id {
            return None;
        }
        self.source.get(span.byte_range())
    }

    /// Index of the last line whose start offset is `<= offset`.
    fn line_index(&self, offset: usize) -> usize {
        match self.line_starts.binary_search(&offset) {
            Ok(idx) => idx,
            Err(idx) => idx.saturating_sub(1),
        }
    }
}

/// Registry of every file loaded during a compilation session.
#[derive(Clone, Debug, Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    /// Creates an empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a file and returns its handle.
    pub fn add(&mut self, name: impl Into<String>, source: impl Into<String>) -> FileId {
        let id = FileId(self.files.len() as u32);
        self.files
            .push(SourceFile::new(id, name.into(), source.into()));
        id
    }

    /// Looks up a registered file.
    pub fn get(&self, id: FileId) -> Option<&SourceFile> {
        self.files.get(id.0 as usize)
    }

    /// Number of registered files.
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Whether no files are registered.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Iterates over all registered files in registration order.
    pub fn iter(&self) -> impl Iterator<Item = &SourceFile> {
        self.files.iter()
    }

    /// Builds a [`Location`] for an offset in a registered file.
    pub fn location(&self, id: FileId, offset: usize) -> Result<Location, CoreError> {
        let file = self.get(id).ok_or(CoreError::UnknownFile(id))?;
        Ok(file.location(offset))
    }

    /// Resolves the text covered by `span`.
    pub fn span_text(&self, span: Span) -> Result<&str, CoreError> {
        let file = self
            .get(span.file())
            .ok_or(CoreError::UnknownFile(span.file()))?;
        file.span_text(span).ok_or(CoreError::SpanOutOfBounds(span))
    }
}

fn compute_line_starts(source: &str) -> Vec<usize> {
    let mut starts = vec![0];
    for (idx, byte) in source.bytes().enumerate() {
        if byte == b'\n' {
            starts.push(idx + 1);
        }
    }
    starts
}

fn trim_line_terminator(source: &str, range: Range<usize>) -> usize {
    let mut end = range.end;
    if end > range.start && source.as_bytes()[end - 1] == b'\n' {
        end -= 1;
    }
    if end > range.start && source.as_bytes()[end - 1] == b'\r' {
        end -= 1;
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_col_tracks_newlines_and_utf8() {
        let mut map = SourceMap::new();
        let id = map.add("t.ape", "ab\nαβγ\ntail");
        let file = map.get(id).unwrap();
        assert_eq!(file.line_col(0), (1, 1));
        assert_eq!(file.line_col(3), (2, 1)); // 'α'
        assert_eq!(file.line_col(5), (2, 2)); // 'β' — one char, two bytes
        assert_eq!(file.line_col(10), (3, 1)); // 't'
                                               // offsets inside a multi-byte character snap to its start
        assert_eq!(file.line_col(4), (2, 1));
        assert_eq!(file.line_count(), 3);
    }

    #[test]
    fn offsets_beyond_eof_are_clamped() {
        let mut map = SourceMap::new();
        let id = map.add("t.ape", "abc");
        let file = map.get(id).unwrap();
        assert_eq!(file.line_col(99), (1, 4));
    }

    #[test]
    fn line_text_strips_terminators() {
        let mut map = SourceMap::new();
        let id = map.add("t.ape", "one\r\ntwo\nthree");
        let file = map.get(id).unwrap();
        assert_eq!(file.line_text(1), Some("one"));
        assert_eq!(file.line_text(2), Some("two"));
        assert_eq!(file.line_text(3), Some("three"));
        assert_eq!(file.line_text(4), None);
    }

    #[test]
    fn span_text_resolves_and_validates() {
        let mut map = SourceMap::new();
        let id = map.add("t.ape", "let x = 1");
        let start = map.location(id, 4).unwrap();
        let end = map.location(id, 5).unwrap();
        assert_eq!(map.span_text(Span::new(start, end)).unwrap(), "x");
        assert!(map.location(FileId(9), 0).is_err());
    }

    #[test]
    fn empty_map_is_empty() {
        let map = SourceMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }
}

//! Byte-accurate source positions shared across the compiler pipeline.

use std::fmt;

use crate::source::FileId;

/// A single point inside a source file.
///
/// `offset` is a UTF-8 byte index, while `line` and `column` are
/// 1-based human readable coordinates computed when the location was
/// produced (the lexer tracks them incrementally while scanning).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Location {
    /// File this location refers to.
    pub file: FileId,
    /// UTF-8 byte offset from the start of the file.
    pub offset: u32,
    /// 1-based line number.
    pub line: u32,
    /// 1-based column number, counted in characters.
    pub column: u32,
}

impl Location {
    /// Creates a location with explicit coordinates.
    pub fn new(file: FileId, offset: u32, line: u32, column: u32) -> Self {
        Self {
            file,
            offset,
            line,
            column,
        }
    }

    /// A zero-length location at the very beginning of `file`.
    pub fn start_of(file: FileId) -> Self {
        Self::new(file, 0, 1, 1)
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// A half-open source range `[start, end)`.
///
/// Every AST node, token, and diagnostic carries a span so that error
/// messages can point at the exact source text that produced them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Span {
    /// Inclusive start of the range.
    pub start: Location,
    /// Exclusive end of the range.
    pub end: Location,
}

impl Span {
    /// Builds a span from its endpoints.
    pub fn new(start: Location, end: Location) -> Self {
        Self { start, end }
    }

    /// A zero-length span located at the beginning of `file`.
    pub fn empty(file: FileId) -> Self {
        let loc = Location::start_of(file);
        Self::new(loc, loc)
    }

    /// File this span belongs to.
    pub fn file(&self) -> FileId {
        self.start.file
    }

    /// Whether the span covers no source text.
    pub fn is_empty(&self) -> bool {
        self.start.offset >= self.end.offset
    }

    /// Smallest span covering both inputs.
    ///
    /// When the two spans belong to different files the left-hand side
    /// wins; cross-file merges only occur in synthetic nodes where the
    /// choice is arbitrary anyway.
    pub fn merge(self, other: Span) -> Span {
        if self.file() != other.file() {
            return self;
        }
        let start = if self.start.offset <= other.start.offset {
            self.start
        } else {
            other.start
        };
        let end = if self.end.offset >= other.end.offset {
            self.end
        } else {
            other.end
        };
        Span::new(start, end)
    }

    /// UTF-8 byte range of this span within its file.
    pub fn byte_range(&self) -> std::ops::Range<usize> {
        self.start.offset as usize..self.end.offset as usize
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start.line == self.end.line {
            write!(
                f,
                "{}:{}:{}-{}",
                self.start.file, self.start.line, self.start.column, self.end.column
            )
        } else {
            write!(
                f,
                "{}:{}:{}-{}:{}",
                self.start.file, self.start.line, self.start.column, self.end.line, self.end.column
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(file: u32, offset: u32, line: u32, column: u32) -> Location {
        Location::new(FileId(file), offset, line, column)
    }

    #[test]
    fn empty_span_is_empty() {
        assert!(Span::empty(FileId(0)).is_empty());
    }

    #[test]
    fn merge_takes_widest_range() {
        let a = Span::new(loc(0, 5, 1, 6), loc(0, 10, 1, 11));
        let b = Span::new(loc(0, 2, 1, 3), loc(0, 7, 1, 8));
        let m = a.merge(b);
        assert_eq!(m.start.offset, 2);
        assert_eq!(m.end.offset, 10);
        assert_eq!(a.merge(b), b.merge(a));
    }

    #[test]
    fn merge_keeps_left_file_on_mismatch() {
        let a = Span::new(loc(1, 0, 1, 1), loc(1, 5, 1, 6));
        let b = Span::new(loc(2, 0, 1, 1), loc(2, 9, 1, 10));
        assert_eq!(a.merge(b).file(), FileId(1));
    }

    #[test]
    fn display_single_line_and_multi_line() {
        let single = Span::new(loc(0, 3, 4, 5), loc(0, 8, 4, 10));
        assert_eq!(single.to_string(), "0:4:5-10");
        let multi = Span::new(loc(0, 3, 4, 5), loc(0, 40, 6, 2));
        assert_eq!(multi.to_string(), "0:4:5-6:2");
    }

    #[test]
    fn byte_range_is_half_open() {
        let s = Span::new(loc(0, 2, 1, 3), loc(0, 7, 1, 8));
        assert_eq!(s.byte_range(), 2..7);
    }
}

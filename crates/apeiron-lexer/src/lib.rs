//! Lexical analyzer for Apeiron source files.
//!
//! The scanner turns UTF-8 source text into a flat [`Vec`] of
//! [`Token`]s terminated by a single [`TokenKind::Eof`]. It tracks
//! 1-based line/column coordinates incrementally so that every token
//! carries a byte-accurate [`Span`] without a second pass over the file.
//!
//! ```no_run
//! use apeiron_core::source::FileId;
//! use apeiron_lexer::Lexer;
//!
//! let tokens = Lexer::new("tensor x : f32[3] = [1.0, 2.0, 3.0]", FileId(0))
//!     .tokenize()
//!     .expect("valid source");
//! ```

pub mod keyword;
pub mod token;

use std::fmt;

use apeiron_core::diagnostic::Diagnostic;
use apeiron_core::source::FileId;
use apeiron_core::span::{Location, Span};

pub use keyword::Keyword;
pub use token::{Token, TokenKind};

/// A lexical failure anchored to the offending source range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LexError {
    /// Explanation shown to the user.
    pub message: String,
    /// Range of source text that could not be tokenized.
    pub span: Span,
}

impl LexError {
    /// Builds an error for `span`.
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (at {})", self.message, self.span)
    }
}

impl std::error::Error for LexError {}

impl From<LexError> for Diagnostic {
    fn from(error: LexError) -> Self {
        Diagnostic::error(error.message)
            .with_code("LEX0001")
            .with_span(error.span)
    }
}

/// Streaming tokenizer over a single source file.
#[derive(Debug, Clone)]
pub struct Lexer<'a> {
    src: &'a str,
    file: FileId,
    pos: usize,
    line: u32,
    column: u32,
}

impl<'a> Lexer<'a> {
    /// Creates a lexer positioned at the start of `src`.
    pub fn new(src: &'a str, file: FileId) -> Self {
        Self {
            src,
            file,
            pos: 0,
            line: 1,
            column: 1,
        }
    }

    /// Consumes the entire input, returning all tokens including the
    /// trailing [`TokenKind::Eof`].
    pub fn tokenize(mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token()?;
            let done = token.kind == TokenKind::Eof;
            tokens.push(token);
            if done {
                return Ok(tokens);
            }
        }
    }

    fn next_token(&mut self) -> Result<Token, LexError> {
        self.skip_trivia()?;
        let start = self.loc();
        let Some(c) = self.peek() else {
            return Ok(Token::eof(Span::new(start, start)));
        };
        let kind = if c.is_ascii_digit() {
            self.number(start)?
        } else if c == '"' {
            self.string(start)?
        } else if is_ident_start(c) {
            self.ident(start)
        } else {
            self.symbol(start)?
        };
        Ok(Token::new(kind, Span::new(start, self.loc())))
    }

    // ── trivia ──────────────────────────────────────────────────────

    fn skip_trivia(&mut self) -> Result<(), LexError> {
        loop {
            match (self.peek(), self.peek_second()) {
                (Some(c), _) if c.is_whitespace() => {
                    self.bump();
                }
                (Some('/'), Some('/')) => {
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                (Some('/'), Some('*')) => {
                    let start = self.loc();
                    self.bump();
                    self.bump();
                    let mut depth = 1usize;
                    while depth > 0 {
                        match (self.peek(), self.peek_second()) {
                            (Some('*'), Some('/')) => {
                                self.bump();
                                self.bump();
                                depth -= 1;
                            }
                            (Some('/'), Some('*')) => {
                                self.bump();
                                self.bump();
                                depth += 1;
                            }
                            (Some(_), _) => {
                                self.bump();
                            }
                            (None, _) => {
                                let span = Span::new(start, self.loc());
                                return Err(LexError::new("unterminated block comment", span));
                            }
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    // ── literals ────────────────────────────────────────────────────

    fn number(&mut self, start: Location) -> Result<TokenKind, LexError> {
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.bump();
        }
        let mut is_float = false;
        if self.peek() == Some('.') && self.peek_second().is_some_and(|c| c.is_ascii_digit()) {
            is_float = true;
            self.bump();
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.bump();
            }
        }
        if self.exponent_ahead() {
            is_float = true;
            self.bump(); // 'e' / 'E'
            if matches!(self.peek(), Some('+' | '-')) {
                self.bump();
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.bump();
            }
        }
        let text = &self.src[start.offset as usize..self.pos];
        if is_float {
            let value = text.parse::<f64>().map_err(|_| {
                LexError::new(
                    format!("invalid floating point literal `{text}`"),
                    Span::new(start, self.loc()),
                )
            })?;
            Ok(TokenKind::Float(value))
        } else {
            let value = text.parse::<i64>().map_err(|_| {
                LexError::new(
                    format!("integer literal `{text}` is out of range"),
                    Span::new(start, self.loc()),
                )
            })?;
            Ok(TokenKind::Integer(value))
        }
    }

    fn exponent_ahead(&self) -> bool {
        if !matches!(self.peek(), Some('e' | 'E')) {
            return false;
        }
        let mut chars = self.src[self.pos + 1..].chars();
        match chars.next() {
            Some(c) if c.is_ascii_digit() => true,
            Some('+' | '-') => chars.next().is_some_and(|c| c.is_ascii_digit()),
            _ => false,
        }
    }

    fn string(&mut self, start: Location) -> Result<TokenKind, LexError> {
        self.bump(); // opening quote
        let mut text = String::new();
        loop {
            match self.peek() {
                None | Some('\n') => {
                    let span = Span::new(start, self.loc());
                    return Err(LexError::new("unterminated string literal", span));
                }
                Some('"') => {
                    self.bump();
                    return Ok(TokenKind::Str(text));
                }
                Some('\\') => {
                    self.bump();
                    let Some(esc) = self.peek() else {
                        let span = Span::new(start, self.loc());
                        return Err(LexError::new("unterminated escape sequence", span));
                    };
                    self.bump();
                    match esc {
                        'n' => text.push('\n'),
                        't' => text.push('\t'),
                        'r' => text.push('\r'),
                        '0' => text.push('\0'),
                        '\\' => text.push('\\'),
                        '"' => text.push('"'),
                        '\'' => text.push('\''),
                        other => {
                            let span = Span::new(start, self.loc());
                            return Err(LexError::new(
                                format!("unknown escape sequence `\\{other}`"),
                                span,
                            ));
                        }
                    }
                }
                Some(c) => {
                    text.push(c);
                    self.bump();
                }
            }
        }
    }

    fn ident(&mut self, start: Location) -> TokenKind {
        while self.peek().is_some_and(is_ident_continue) {
            self.bump();
        }
        let text = &self.src[start.offset as usize..self.pos];
        match Keyword::from_ident(text) {
            Some(keyword) => TokenKind::Keyword(keyword),
            None => TokenKind::Ident(text.to_string()),
        }
    }

    fn symbol(&mut self, start: Location) -> Result<TokenKind, LexError> {
        let c = self.peek().expect("symbol called with non-empty input");
        let kind = match c {
            '(' => {
                self.bump();
                TokenKind::LParen
            }
            ')' => {
                self.bump();
                TokenKind::RParen
            }
            '{' => {
                self.bump();
                TokenKind::LBrace
            }
            '}' => {
                self.bump();
                TokenKind::RBrace
            }
            '[' => {
                self.bump();
                TokenKind::LBracket
            }
            ']' => {
                self.bump();
                TokenKind::RBracket
            }
            ',' => {
                self.bump();
                TokenKind::Comma
            }
            ':' => {
                self.bump();
                TokenKind::Colon
            }
            ';' => {
                self.bump();
                TokenKind::Semi
            }
            '.' => {
                self.bump();
                TokenKind::Dot
            }
            '+' => {
                self.bump();
                TokenKind::Plus
            }
            '*' => {
                self.bump();
                TokenKind::Star
            }
            '/' => {
                self.bump();
                TokenKind::Slash
            }
            '%' => {
                self.bump();
                TokenKind::Percent
            }
            '@' => {
                self.bump();
                TokenKind::At
            }
            '-' => {
                self.bump();
                if self.peek() == Some('>') {
                    self.bump();
                    TokenKind::Arrow
                } else {
                    TokenKind::Minus
                }
            }
            '=' => {
                self.bump();
                match self.peek() {
                    Some('=') => {
                        self.bump();
                        TokenKind::EqEq
                    }
                    Some('>') => {
                        self.bump();
                        TokenKind::FatArrow
                    }
                    _ => TokenKind::Assign,
                }
            }
            '!' => {
                self.bump();
                if self.peek() == Some('=') {
                    self.bump();
                    TokenKind::NotEq
                } else {
                    TokenKind::Bang
                }
            }
            '<' => {
                self.bump();
                if self.peek() == Some('=') {
                    self.bump();
                    TokenKind::Le
                } else {
                    TokenKind::Lt
                }
            }
            '>' => {
                self.bump();
                if self.peek() == Some('=') {
                    self.bump();
                    TokenKind::Ge
                } else {
                    TokenKind::Gt
                }
            }
            other => {
                let span = Span::new(start, {
                    self.bump();
                    self.loc()
                });
                return Err(LexError::new(
                    format!("unexpected character `{other}`"),
                    span,
                ));
            }
        };
        Ok(kind)
    }

    // ── cursor primitives ───────────────────────────────────────────

    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn peek_second(&self) -> Option<char> {
        self.src[self.pos..].chars().nth(1)
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c)
    }

    fn loc(&self) -> Location {
        Location::new(self.file, self.pos as u32, self.line, self.column)
    }
}

fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_alphabetic()
}

fn is_ident_continue(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(source: &str) -> Vec<Token> {
        Lexer::new(source, FileId(0))
            .tokenize()
            .expect("lexing failed")
    }

    fn kinds(source: &str) -> Vec<TokenKind> {
        lex(source).into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn empty_input_yields_only_eof() {
        let tokens = lex("");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
        assert_eq!(tokens[0].span.start.line, 1);
        assert_eq!(tokens[0].span.start.column, 1);
    }

    #[test]
    fn all_keywords_lex_as_keywords() {
        for kw in Keyword::ALL {
            assert_eq!(
                kinds(kw.as_str()),
                vec![TokenKind::Keyword(kw), TokenKind::Eof],
                "keyword `{}` failed to lex",
                kw.as_str()
            );
        }
    }

    #[test]
    fn identifiers_vs_keywords() {
        assert_eq!(
            kinds("ShortTermMem with_respect_toX tensorx"),
            vec![
                TokenKind::Ident("ShortTermMem".into()),
                TokenKind::Ident("with_respect_toX".into()),
                TokenKind::Ident("tensorx".into()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn unicode_identifiers() {
        assert_eq!(
            kinds("намерение"),
            vec![TokenKind::Ident("намерение".into()), TokenKind::Eof]
        );
    }

    #[test]
    fn integer_and_float_literals() {
        assert_eq!(
            kinds("42 0.5 85.5 1e-3 2E+4 0.001"),
            vec![
                TokenKind::Integer(42),
                TokenKind::Float(0.5),
                TokenKind::Float(85.5),
                TokenKind::Float(1e-3),
                TokenKind::Float(2e4),
                TokenKind::Float(0.001),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn trailing_dot_is_not_a_float() {
        // A dot only continues a number when directly followed by a digit.
        assert_eq!(
            kinds("1. rest"),
            vec![
                TokenKind::Integer(1),
                TokenKind::Dot,
                TokenKind::Ident("rest".into()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn string_escapes_are_decoded() {
        assert_eq!(
            kinds(r#""a\nb\"c\\d""#),
            vec![TokenKind::Str("a\nb\"c\\d".into()), TokenKind::Eof]
        );
    }

    #[test]
    fn unterminated_string_is_an_error() {
        let err = Lexer::new("\"abc", FileId(0)).tokenize().unwrap_err();
        assert_eq!(err.message, "unterminated string literal");
        assert_eq!(err.span.start.column, 1);
    }

    #[test]
    fn newline_ends_string() {
        let err = Lexer::new("\"abc\ndef\"", FileId(0))
            .tokenize()
            .unwrap_err();
        assert_eq!(err.message, "unterminated string literal");
    }

    #[test]
    fn unknown_escape_is_rejected() {
        let err = Lexer::new(r#""a\qb""#, FileId(0)).tokenize().unwrap_err();
        assert!(err.message.contains("unknown escape sequence"));
    }

    #[test]
    fn integer_overflow_is_rejected() {
        let err = Lexer::new("99999999999999999999999", FileId(0))
            .tokenize()
            .unwrap_err();
        assert!(err.message.contains("out of range"));
    }

    #[test]
    fn multi_char_operators() {
        assert_eq!(
            kinds("-> => == != <= >= @ + -"),
            vec![
                TokenKind::Arrow,
                TokenKind::FatArrow,
                TokenKind::EqEq,
                TokenKind::NotEq,
                TokenKind::Le,
                TokenKind::Ge,
                TokenKind::At,
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn comments_are_skipped() {
        assert_eq!(
            kinds("let // trailing\n/* block\nnested /* inner */ end */ x"),
            vec![
                TokenKind::Keyword(Keyword::Let),
                TokenKind::Ident("x".into()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn unterminated_block_comment_is_an_error() {
        let err = Lexer::new("/* never closed", FileId(0))
            .tokenize()
            .unwrap_err();
        assert_eq!(err.message, "unterminated block comment");
    }

    #[test]
    fn unexpected_character_is_an_error() {
        let err = Lexer::new("a # b", FileId(0)).tokenize().unwrap_err();
        assert_eq!(err.message, "unexpected character `#`");
        assert_eq!(err.span.start.column, 3);
        assert_eq!(err.span.end.column, 4);
    }

    #[test]
    fn spans_track_lines_and_columns() {
        let tokens = lex("let x = 1\nlet y = 2");
        let lines: Vec<(u32, u32)> = tokens
            .iter()
            .map(|t| (t.span.start.line, t.span.start.column))
            .collect();
        assert_eq!(
            lines,
            vec![
                (1, 1),
                (1, 5),
                (1, 7),
                (1, 9),
                (2, 1),
                (2, 5),
                (2, 7),
                (2, 9),
                (2, 10)
            ]
        );
    }

    #[test]
    fn spans_cover_exact_source_text() {
        let source = "tensor W";
        let tokens = lex(source);
        assert_eq!(tokens[0].span.byte_range(), 0..6);
        assert_eq!(&source[tokens[0].span.byte_range()], "tensor");
        assert_eq!(&source[tokens[1].span.byte_range()], "W");
    }

    #[test]
    fn lex_error_converts_to_diagnostic() {
        let err = LexError::new("boom", Span::empty(FileId(2)));
        let diag: Diagnostic = err.into();
        assert_eq!(diag.code.as_deref(), Some("LEX0001"));
        assert_eq!(diag.span, Some(Span::empty(FileId(2))));
    }

    #[test]
    fn spec_snippet_lexes_cleanly() {
        let source = r#"
            tensor y = (W @ x) + b
            gradient loss with_respect_to W
            optimizer adam { lr: 0.001, beta1: 0.9 }
            train model on dataset_ref
        "#;
        let tokens = lex(source);
        assert_eq!(tokens.last().unwrap().kind, TokenKind::Eof);
        assert!(tokens.len() >= 30);
    }
}

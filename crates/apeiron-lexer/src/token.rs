//! Token vocabulary produced by the lexer.

use std::fmt;

use apeiron_core::span::Span;

use crate::keyword::Keyword;

/// Lexical class of a token.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    /// Integer literal such as `42`.
    Integer(i64),
    /// Floating point literal such as `0.001`.
    Float(f64),
    /// String literal, escapes already decoded.
    Str(String),
    /// Non-reserved identifier such as `ShortTermMem`.
    Ident(String),
    /// Reserved word.
    Keyword(Keyword),

    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `,`
    Comma,
    /// `:`
    Colon,
    /// `;`
    Semi,
    /// `.`
    Dot,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `%`
    Percent,
    /// `@` — matrix multiplication.
    At,
    /// `=`
    Assign,
    /// `==`
    EqEq,
    /// `!=`
    NotEq,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `->` — arrow used by causal edges and consolidation.
    Arrow,
    /// `=>`
    FatArrow,
    /// `!`
    Bang,
    /// `?` — the `try` propagation postfix.
    Question,
    /// End of input marker always emitted last.
    Eof,
}

impl TokenKind {
    /// Human readable description used inside parser errors.
    pub fn describe(&self) -> String {
        match self {
            TokenKind::Integer(value) => format!("integer literal `{value}`"),
            TokenKind::Float(value) => format!("float literal `{value:?}`"),
            TokenKind::Str(text) => format!("string literal `\"{text}\"`"),
            TokenKind::Ident(name) => format!("identifier `{name}`"),
            TokenKind::Keyword(keyword) => format!("keyword `{keyword}`"),
            TokenKind::Eof => "end of file".to_string(),
            other => format!("`{other}`"),
        }
    }

    /// Whether this token can begin an expression.
    pub fn starts_expr(&self) -> bool {
        matches!(
            self,
            TokenKind::Integer(_)
                | TokenKind::Float(_)
                | TokenKind::Str(_)
                | TokenKind::Ident(_)
                | TokenKind::LParen
                | TokenKind::LBracket
                | TokenKind::LBrace
                | TokenKind::Minus
                | TokenKind::Bang
                | TokenKind::Keyword(Keyword::True)
                | TokenKind::Keyword(Keyword::False)
                | TokenKind::Keyword(Keyword::If)
        )
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Integer(value) => write!(f, "{value}"),
            TokenKind::Float(value) => write!(f, "{value:?}"),
            TokenKind::Str(text) => write!(f, "\"{text}\""),
            TokenKind::Ident(name) => f.write_str(name),
            TokenKind::Keyword(keyword) => write!(f, "{keyword}"),
            TokenKind::LParen => f.write_str("("),
            TokenKind::RParen => f.write_str(")"),
            TokenKind::LBrace => f.write_str("{"),
            TokenKind::RBrace => f.write_str("}"),
            TokenKind::LBracket => f.write_str("["),
            TokenKind::RBracket => f.write_str("]"),
            TokenKind::Comma => f.write_str(","),
            TokenKind::Colon => f.write_str(":"),
            TokenKind::Semi => f.write_str(";"),
            TokenKind::Dot => f.write_str("."),
            TokenKind::Plus => f.write_str("+"),
            TokenKind::Minus => f.write_str("-"),
            TokenKind::Star => f.write_str("*"),
            TokenKind::Slash => f.write_str("/"),
            TokenKind::Percent => f.write_str("%"),
            TokenKind::At => f.write_str("@"),
            TokenKind::Assign => f.write_str("="),
            TokenKind::EqEq => f.write_str("=="),
            TokenKind::NotEq => f.write_str("!="),
            TokenKind::Lt => f.write_str("<"),
            TokenKind::Le => f.write_str("<="),
            TokenKind::Gt => f.write_str(">"),
            TokenKind::Ge => f.write_str(">="),
            TokenKind::Arrow => f.write_str("->"),
            TokenKind::FatArrow => f.write_str("=>"),
            TokenKind::Bang => f.write_str("!"),
            TokenKind::Question => f.write_str("?"),
            TokenKind::Eof => f.write_str("end of file"),
        }
    }
}

/// A token together with the source range it consumed.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    /// Lexical class.
    pub kind: TokenKind,
    /// Source range `[start, end)` covered by this token.
    pub span: Span,
}

impl Token {
    /// Builds a token.
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }

    /// The synthetic end-of-file token.
    pub fn eof(span: Span) -> Self {
        Self::new(TokenKind::Eof, span)
    }

    /// Human readable description for error messages.
    pub fn describe(&self) -> String {
        self.kind.describe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apeiron_core::source::FileId;

    fn span() -> Span {
        Span::empty(FileId(0))
    }

    #[test]
    fn describe_distinguishes_token_classes() {
        assert_eq!(TokenKind::Integer(7).describe(), "integer literal `7`");
        assert_eq!(TokenKind::Ident("x".into()).describe(), "identifier `x`");
        assert_eq!(TokenKind::Keyword(Keyword::Let).describe(), "keyword `let`");
        assert_eq!(TokenKind::Arrow.describe(), "`->`");
        assert_eq!(TokenKind::Eof.describe(), "end of file");
    }

    #[test]
    fn starts_expr_covers_literal_and_delimiters() {
        assert!(TokenKind::Integer(1).starts_expr());
        assert!(TokenKind::Ident("x".into()).starts_expr());
        assert!(TokenKind::LParen.starts_expr());
        assert!(TokenKind::Minus.starts_expr());
        assert!(TokenKind::Keyword(Keyword::If).starts_expr());
        assert!(!TokenKind::Keyword(Keyword::Let).starts_expr());
        assert!(!TokenKind::Comma.starts_expr());
        assert!(!TokenKind::Eof.starts_expr());
    }

    #[test]
    fn display_renders_source_spelling() {
        assert_eq!(TokenKind::At.to_string(), "@");
        assert_eq!(TokenKind::Str("hi".into()).to_string(), "\"hi\"");
        assert_eq!(Token::eof(span()).describe(), "end of file");
    }
}

//! Pratt expression parser.
//!
//! Binding powers are derived from [`BinaryOp::precedence`] so the parser
//! and the AST pretty printer can never disagree about associativity.

use apeiron_ast::{Arm, BinaryOp, Expr, Ident, Literal, Pattern, UnaryOp};
use apeiron_lexer::{Keyword, Token, TokenKind};

use crate::error::ParseError;
use crate::parser::Parser;

impl Parser {
    /// Parses an expression, stopping once the next operator binds less
    /// tightly than `min_bp`.
    pub fn parse_expr(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_prefix()?;
        loop {
            // Postfix operators bind tighter than every infix operator.
            match self.peek_kind() {
                TokenKind::LParen => {
                    lhs = self.parse_call(lhs)?;
                    continue;
                }
                TokenKind::LBracket => {
                    lhs = self.parse_index(lhs)?;
                    continue;
                }
                TokenKind::Dot => {
                    lhs = self.parse_field(lhs)?;
                    continue;
                }
                TokenKind::Question => {
                    lhs = self.parse_try(lhs)?;
                    continue;
                }
                _ => {}
            }
            let Some((left_bp, right_bp, op)) = infix_binding(self.peek_kind()) else {
                break;
            };
            if left_bp < min_bp {
                break;
            }
            self.bump();
            let rhs = self.parse_expr(right_bp)?;
            let span = lhs.span().merge(rhs.span());
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    pub(crate) fn parse_prefix(&mut self) -> Result<Expr, ParseError> {
        let token = self.peek().clone();
        match token.kind {
            TokenKind::Integer(value) => {
                self.bump();
                Ok(Expr::Literal {
                    value: Literal::Int(value),
                    span: token.span,
                })
            }
            TokenKind::Float(value) => {
                self.bump();
                Ok(Expr::Literal {
                    value: Literal::Float(value),
                    span: token.span,
                })
            }
            TokenKind::Str(value) => {
                self.bump();
                Ok(Expr::Literal {
                    value: Literal::Str(value),
                    span: token.span,
                })
            }
            TokenKind::Keyword(Keyword::True) => {
                self.bump();
                Ok(Expr::Literal {
                    value: Literal::Bool(true),
                    span: token.span,
                })
            }
            TokenKind::Keyword(Keyword::False) => {
                self.bump();
                Ok(Expr::Literal {
                    value: Literal::Bool(false),
                    span: token.span,
                })
            }
            TokenKind::Keyword(Keyword::If) => self.parse_if(),
            TokenKind::Keyword(Keyword::Match) => self.parse_match(),
            TokenKind::Ident(_) | TokenKind::Keyword(_) => Ok(Expr::Ident(self.parse_ident()?)),
            TokenKind::Minus => {
                self.bump();
                let operand = self.parse_expr(UnaryOp::Neg.precedence() + 1)?;
                let span = token.span.merge(operand.span());
                Ok(Expr::Unary {
                    op: UnaryOp::Neg,
                    operand: Box::new(operand),
                    span,
                })
            }
            TokenKind::Bang => {
                self.bump();
                let operand = self.parse_expr(UnaryOp::Not.precedence() + 1)?;
                let span = token.span.merge(operand.span());
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(operand),
                    span,
                })
            }
            TokenKind::LParen => {
                self.bump();
                let inner = self.parse_expr(0)?;
                self.expect(&TokenKind::RParen)?;
                Ok(inner)
            }
            TokenKind::LBracket => {
                let open = self.bump();
                let mut elements = Vec::new();
                if !self.at(&TokenKind::RBracket) {
                    loop {
                        elements.push(self.parse_expr(0)?);
                        if !self.eat(&TokenKind::Comma) {
                            break;
                        }
                        if self.at(&TokenKind::RBracket) {
                            break;
                        }
                    }
                }
                let close = self.expect(&TokenKind::RBracket)?;
                Ok(Expr::Array {
                    elements,
                    span: open.span.merge(close.span),
                })
            }
            TokenKind::LBrace => {
                let open = self.bump();
                self.parse_block_body(open)
            }
            _ => Err(self.unexpected("an expression")),
        }
    }

    fn parse_call(&mut self, callee: Expr) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::LParen)?;
        let mut args = Vec::new();
        if !self.at(&TokenKind::RParen) {
            loop {
                args.push(self.parse_expr(0)?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
                if self.at(&TokenKind::RParen) {
                    break;
                }
            }
        }
        let close = self.expect(&TokenKind::RParen)?;
        let span = callee.span().merge(close.span);
        Ok(Expr::Call {
            callee: Box::new(callee),
            args,
            span,
        })
    }

    fn parse_index(&mut self, target: Expr) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::LBracket)?;
        let index = self.parse_expr(0)?;
        let close = self.expect(&TokenKind::RBracket)?;
        let span = target.span().merge(close.span);
        Ok(Expr::Index {
            target: Box::new(target),
            index: Box::new(index),
            span,
        })
    }

    fn parse_field(&mut self, target: Expr) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::Dot)?;
        let field = self.parse_ident()?;
        let span = target.span().merge(field.span);
        Ok(Expr::Field {
            target: Box::new(target),
            field,
            span,
        })
    }

    fn parse_try(&mut self, operand: Expr) -> Result<Expr, ParseError> {
        let question = self.expect(&TokenKind::Question)?;
        let span = operand.span().merge(question.span);
        Ok(Expr::Try {
            operand: Box::new(operand),
            span,
        })
    }

    fn parse_if(&mut self) -> Result<Expr, ParseError> {
        let keyword = self.expect_keyword(Keyword::If)?;
        let condition = self.parse_expr(0)?;
        let then_branch = self.parse_braced_block()?;
        let else_branch = if self.eat_keyword(Keyword::Else) {
            if self.at_keyword(Keyword::If) {
                Some(Box::new(self.parse_if()?))
            } else {
                Some(Box::new(self.parse_braced_block()?))
            }
        } else {
            None
        };
        let span = match &else_branch {
            Some(branch) => keyword.span.merge(branch.span()),
            None => keyword.span.merge(then_branch.span()),
        };
        Ok(Expr::If {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch,
            span,
        })
    }

    /// Parses `match <expr> { <pattern> [if <guard>] => <body>, ... }`.
    fn parse_match(&mut self) -> Result<Expr, ParseError> {
        let keyword = self.expect_keyword(Keyword::Match)?;
        let scrutinee = self.parse_expr(0)?;
        self.expect(&TokenKind::LBrace)?;
        let mut arms = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            let arm_start = self.peek().span;
            let pat = self.parse_pattern()?;
            let guard = if self.eat_keyword(Keyword::If) {
                Some(self.parse_expr(0)?)
            } else {
                None
            };
            self.expect(&TokenKind::FatArrow)?;
            let body = self.parse_expr(0)?;
            let span = arm_start.merge(body.span());
            arms.push(Arm::new(pat, guard, body, span));
            if !self.eat(&TokenKind::Comma) {
                break;
            }
            if self.at(&TokenKind::RBrace) {
                break;
            }
        }
        let close = self.expect(&TokenKind::RBrace)?;
        Ok(Expr::Match {
            expr: Box::new(scrutinee),
            arms,
            span: keyword.span.merge(close.span),
        })
    }

    /// Parses a single `match` arm pattern.
    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        let token = self.peek().clone();
        match &token.kind {
            TokenKind::Integer(value) => {
                let value = *value;
                self.bump();
                Ok(Pattern::Literal(Literal::Int(value)))
            }
            TokenKind::Float(value) => {
                let value = *value;
                self.bump();
                Ok(Pattern::Literal(Literal::Float(value)))
            }
            TokenKind::Str(value) => {
                let value = value.clone();
                self.bump();
                Ok(Pattern::Literal(Literal::Str(value)))
            }
            TokenKind::Keyword(Keyword::True) => {
                self.bump();
                Ok(Pattern::Literal(Literal::Bool(true)))
            }
            TokenKind::Keyword(Keyword::False) => {
                self.bump();
                Ok(Pattern::Literal(Literal::Bool(false)))
            }
            TokenKind::Minus => {
                self.bump();
                let number = self.peek().clone();
                match number.kind {
                    TokenKind::Integer(value) => {
                        self.bump();
                        Ok(Pattern::Literal(Literal::Int(-value)))
                    }
                    TokenKind::Float(value) => {
                        self.bump();
                        Ok(Pattern::Literal(Literal::Float(-value)))
                    }
                    _ => Err(self.unexpected("a number")),
                }
            }
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.bump();
                if name == "_" {
                    Ok(Pattern::Wildcard)
                } else if name == "None" {
                    Ok(Pattern::Variant {
                        name: Ident::new(&name, token.span),
                        inner: None,
                    })
                } else if matches!(name.as_str(), "Some" | "Ok" | "Err")
                    && self.at(&TokenKind::LParen)
                {
                    self.bump();
                    let inner = self.parse_pattern()?;
                    self.expect(&TokenKind::RParen)?;
                    Ok(Pattern::Variant {
                        name: Ident::new(&name, token.span),
                        inner: Some(Box::new(inner)),
                    })
                } else {
                    Ok(Pattern::Bind(Ident::new(&name, token.span)))
                }
            }
            TokenKind::LParen => {
                self.bump();
                let mut elements = Vec::new();
                if !self.at(&TokenKind::RParen) {
                    loop {
                        elements.push(self.parse_pattern()?);
                        if !self.eat(&TokenKind::Comma) {
                            break;
                        }
                        if self.at(&TokenKind::RParen) {
                            break;
                        }
                    }
                }
                self.expect(&TokenKind::RParen)?;
                if elements.len() == 1 {
                    Ok(elements.pop().expect("single element checked above"))
                } else {
                    Ok(Pattern::Tuple(elements))
                }
            }
            _ => Err(self.unexpected("a pattern")),
        }
    }

    /// Parses a `{ ... }` block, returning it as an expression node.
    pub(crate) fn parse_braced_block(&mut self) -> Result<Expr, ParseError> {
        let open = self.expect(&TokenKind::LBrace)?;
        self.parse_block_body(open)
    }

    fn parse_block_body(&mut self, open: Token) -> Result<Expr, ParseError> {
        let mut items = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            items.push(self.parse_item()?);
        }
        let close = self.expect(&TokenKind::RBrace)?;
        Ok(Expr::Block {
            items,
            span: open.span.merge(close.span),
        })
    }
}

/// Binding powers for every infix operator: `(left, right, op)`.
///
/// `=` is right associative (`right == left`), all other operators are
/// left associative (`right == left + 1`).
fn infix_binding(kind: &TokenKind) -> Option<(u8, u8, BinaryOp)> {
    let op = match kind {
        TokenKind::Assign => BinaryOp::Assign,
        TokenKind::EqEq => BinaryOp::Eq,
        TokenKind::NotEq => BinaryOp::Ne,
        TokenKind::Lt => BinaryOp::Lt,
        TokenKind::Le => BinaryOp::Le,
        TokenKind::Gt => BinaryOp::Gt,
        TokenKind::Ge => BinaryOp::Ge,
        TokenKind::Plus => BinaryOp::Add,
        TokenKind::Minus => BinaryOp::Sub,
        TokenKind::Star => BinaryOp::Mul,
        TokenKind::Slash => BinaryOp::Div,
        TokenKind::Percent => BinaryOp::Rem,
        TokenKind::At => BinaryOp::MatMul,
        _ => return None,
    };
    let bp = op.precedence();
    Some(if matches!(op, BinaryOp::Assign) {
        (bp, bp, op)
    } else {
        (bp, bp + 1, op)
    })
}

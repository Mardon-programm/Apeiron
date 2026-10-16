//! Recursive descent parser for Apeiron statements and declarations.
//!
//! Two cooperating layers produce the [`Program`]:
//!
//! * this module performs recursive descent over the domain-specific
//!   primitives (`tensor`, `belief`, `cause`, `memory`, `architecture`,
//!   `society`, ...), driving the parser token by token;
//! * [`crate::expr`] implements a Pratt parser used for every expression
//!   that appears inside those constructs.

use apeiron_ast::{
    ArchOp, ArchitectureBlock, BeliefStmt, Field, FnDecl, Ident, Item, LetDecl, MemoryDecl,
    MemoryTier, Param, Program, SocietyBlock, SocietyMember, TensorDecl, TensorKind, TypeDim,
    TypeExpr,
};
use apeiron_core::span::Span;
use apeiron_lexer::{Keyword, Token, TokenKind};

use crate::error::ParseError;

/// Token stream cursor with lookahead plus the whole item grammar.
pub struct Parser {
    pub(crate) tokens: Vec<Token>,
    pub(crate) pos: usize,
}

impl Parser {
    /// Creates a parser over a complete token stream.
    ///
    /// The stream is expected to end with a [`TokenKind::Eof`] token as
    /// produced by [`apeiron_lexer::Lexer::tokenize`]; a missing marker is
    /// tolerated and synthesized on demand.
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Parses a whole file.
    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        let start = self.peek().span;
        let mut items = Vec::new();
        while !self.at(&TokenKind::Eof) {
            items.push(self.parse_item()?);
        }
        let end = self.peek().span;
        Ok(Program::new(items, start.merge(end)))
    }

    // ── statements & declarations ───────────────────────────────────

    /// Parses a single statement or declaration.
    pub fn parse_item(&mut self) -> Result<Item, ParseError> {
        let token = self.peek().clone();
        let item = match &token.kind {
            TokenKind::Keyword(Keyword::Let) => self.parse_let()?,
            TokenKind::Keyword(Keyword::Fn) => Item::Fn(self.parse_fn()?),
            TokenKind::Keyword(Keyword::Return) => self.parse_return()?,
            TokenKind::Keyword(Keyword::While) => self.parse_while()?,
            TokenKind::Keyword(Keyword::For) => self.parse_for()?,
            TokenKind::Keyword(Keyword::Break) => {
                self.bump();
                Item::Break { span: token.span }
            }
            TokenKind::Keyword(Keyword::Continue) => {
                self.bump();
                Item::Continue { span: token.span }
            }
            TokenKind::Keyword(Keyword::Tensor) => self.parse_tensor(TensorKind::Tensor)?,
            TokenKind::Keyword(Keyword::Parameter) => self.parse_tensor(TensorKind::Parameter)?,
            TokenKind::Keyword(Keyword::Optimizer) => self.parse_optimizer()?,
            TokenKind::Keyword(Keyword::Train) => self.parse_train()?,
            TokenKind::Keyword(Keyword::Gradient) => self.parse_gradient()?,
            TokenKind::Keyword(Keyword::Distribution) => self.parse_distribution()?,
            TokenKind::Keyword(Keyword::Belief) => Item::Belief(self.parse_belief()?),
            TokenKind::Keyword(Keyword::Cause) => self.parse_cause()?,
            TokenKind::Keyword(Keyword::Observe) => self.parse_target_assign(true)?,
            TokenKind::Keyword(Keyword::Intervene) => self.parse_target_assign(false)?,
            TokenKind::Keyword(Keyword::Counterfactual) => self.parse_counterfactual()?,
            TokenKind::Keyword(Keyword::Simulate) => self.parse_simulate()?,
            TokenKind::Keyword(Keyword::Memory) => Item::Memory(self.parse_memory()?),
            TokenKind::Keyword(Keyword::Remember) => self.parse_remember()?,
            TokenKind::Keyword(Keyword::Consolidate) => self.parse_consolidate()?,
            TokenKind::Keyword(Keyword::Recall) => self.parse_recall()?,
            TokenKind::Keyword(Keyword::Plan) => self.parse_plan()?,
            TokenKind::Keyword(Keyword::SelfModel) => self.parse_self_model()?,
            TokenKind::Keyword(Keyword::Architecture) => {
                Item::Architecture(self.parse_architecture()?)
            }
            TokenKind::Keyword(Keyword::Society) => Item::Society(self.parse_society()?),
            TokenKind::Keyword(Keyword::Route) => self.parse_route()?,
            _ => Item::Expr(self.parse_expr(0)?),
        };
        // Semicolons are accepted as optional statement terminators.
        while self.eat(&TokenKind::Semi) {}
        Ok(item)
    }

    fn parse_let(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::Let)?;
        let name = self.parse_ident()?;
        let ty = if self.eat(&TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::Assign)?;
        let value = self.parse_expr(0)?;
        let span = keyword.span.merge(value.span());
        Ok(Item::Let(LetDecl {
            name,
            ty,
            value,
            span,
        }))
    }

    fn parse_fn(&mut self) -> Result<FnDecl, ParseError> {
        let keyword = self.expect_keyword(Keyword::Fn)?;
        let name = self.parse_ident()?;
        self.expect(&TokenKind::LParen)?;
        let mut params = Vec::new();
        if !self.at(&TokenKind::RParen) {
            loop {
                let param_name = self.parse_ident()?;
                let ty = if self.eat(&TokenKind::Colon) {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                let param_span = match &ty {
                    Some(ty) => param_name.span.merge(ty.span),
                    None => param_name.span,
                };
                params.push(Param {
                    name: param_name,
                    ty,
                    span: param_span,
                });
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RParen)?;
        let return_type = if self.eat(&TokenKind::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };
        let body = self.parse_braced_block()?;
        let span = keyword.span.merge(body.span());
        Ok(FnDecl {
            name,
            params,
            return_type,
            body,
            span,
        })
    }

    fn parse_return(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::Return)?;
        let value = if self.peek().kind.starts_expr() {
            Some(self.parse_expr(0)?)
        } else {
            None
        };
        let span = match &value {
            Some(value) => keyword.span.merge(value.span()),
            None => keyword.span,
        };
        Ok(Item::Return { value, span })
    }

    fn parse_while(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::While)?;
        let condition = self.parse_expr(0)?;
        let body = self.parse_braced_block()?;
        let span = keyword.span.merge(body.span());
        Ok(Item::While {
            condition,
            body,
            span,
        })
    }

    fn parse_for(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::For)?;
        let variable = self.parse_ident()?;
        self.expect_keyword(Keyword::In)?;
        let iterable = self.parse_expr(0)?;
        let body = self.parse_braced_block()?;
        let span = keyword.span.merge(body.span());
        Ok(Item::For {
            variable,
            iterable,
            body,
            span,
        })
    }

    fn parse_tensor(&mut self, kind: TensorKind) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(match kind {
            TensorKind::Tensor => Keyword::Tensor,
            TensorKind::Parameter => Keyword::Parameter,
        })?;
        let name = self.parse_ident()?;
        let ty = if self.eat(&TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::Assign)?;
        let value = self.parse_expr(0)?;
        let span = keyword.span.merge(value.span());
        Ok(Item::Tensor(TensorDecl {
            kind,
            name,
            ty,
            value,
            span,
        }))
    }

    fn parse_optimizer(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::Optimizer)?;
        let kind = self.parse_ident()?;
        let settings = self.parse_colon_fields()?;
        let span = keyword.span.merge(self.previous_span());
        Ok(Item::Optimizer {
            kind,
            settings,
            span,
        })
    }

    fn parse_train(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::Train)?;
        let model = self.parse_ident()?;
        self.expect_keyword(Keyword::On)?;
        let dataset = self.parse_ident()?;
        let span = keyword.span.merge(dataset.span);
        Ok(Item::Train {
            model,
            dataset,
            span,
        })
    }

    fn parse_gradient(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::Gradient)?;
        let target = self.parse_ident()?;
        self.expect_keyword(Keyword::WithRespectTo)?;
        let with_respect_to = self.parse_ident()?;
        let span = keyword.span.merge(with_respect_to.span);
        Ok(Item::Gradient {
            target,
            with_respect_to,
            span,
        })
    }

    fn parse_distribution(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::Distribution)?;
        let name = self.parse_ident()?;
        let categories = self.parse_colon_fields()?;
        let span = keyword.span.merge(self.previous_span());
        Ok(Item::Distribution {
            name,
            categories,
            span,
        })
    }

    fn parse_belief(&mut self) -> Result<BeliefStmt, ParseError> {
        let keyword = self.expect_keyword(Keyword::Belief)?;
        let name = self.parse_ident()?;
        if self.eat_keyword(Keyword::Update) {
            let fields = self.parse_colon_fields()?;
            let span = keyword.span.merge(self.previous_span());
            Ok(BeliefStmt::Update { name, fields, span })
        } else {
            let fields = self.parse_colon_fields()?;
            let span = keyword.span.merge(self.previous_span());
            Ok(BeliefStmt::Declare { name, fields, span })
        }
    }

    fn parse_cause(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::Cause)?;
        let from = self.parse_ident()?;
        self.expect(&TokenKind::Arrow)?;
        let to = self.parse_ident()?;
        let span = keyword.span.merge(to.span);
        Ok(Item::Cause { from, to, span })
    }

    /// Shared shape for `observe A = value` (`is_observe`) and
    /// `intervene A = value`.
    fn parse_target_assign(&mut self, is_observe: bool) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(if is_observe {
            Keyword::Observe
        } else {
            Keyword::Intervene
        })?;
        let target = self.parse_ident()?;
        self.expect(&TokenKind::Assign)?;
        let value = self.parse_expr(0)?;
        let span = keyword.span.merge(value.span());
        Ok(if is_observe {
            Item::Observe {
                target,
                value,
                span,
            }
        } else {
            Item::Intervene {
                target,
                value,
                span,
            }
        })
    }

    fn parse_counterfactual(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::Counterfactual)?;
        let target = self.parse_ident()?;
        self.expect_keyword(Keyword::If)?;
        let condition = self.parse_expr(0)?;
        let span = keyword.span.merge(condition.span());
        Ok(Item::Counterfactual {
            target,
            condition,
            span,
        })
    }

    fn parse_simulate(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::Simulate)?;
        let parameter = self.parse_ident()?;
        self.expect(&TokenKind::Assign)?;
        let value = self.parse_expr(0)?;
        let span = keyword.span.merge(value.span());
        Ok(Item::Simulate {
            parameter,
            value,
            span,
        })
    }

    fn parse_memory(&mut self) -> Result<MemoryDecl, ParseError> {
        let keyword = self.expect_keyword(Keyword::Memory)?;
        let tier_token = self.peek().clone();
        let tier = match tier_token.kind {
            TokenKind::Keyword(Keyword::Episodic) => MemoryTier::Episodic,
            TokenKind::Keyword(Keyword::Semantic) => MemoryTier::Semantic,
            TokenKind::Keyword(Keyword::Procedural) => MemoryTier::Procedural,
            _ => return Err(self.unexpected("`episodic`, `semantic`, or `procedural`")),
        };
        self.bump();
        let name = self.parse_ident()?;
        let fields = self.parse_colon_fields()?;
        let span = keyword.span.merge(self.previous_span());
        Ok(MemoryDecl {
            tier,
            name,
            fields,
            span,
        })
    }

    fn parse_remember(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::Remember)?;
        let memory = self.parse_ident()?;
        let entries = self.parse_eq_fields(true)?;
        let span = keyword.span.merge(self.previous_span());
        Ok(Item::Remember {
            memory,
            entries,
            span,
        })
    }

    fn parse_consolidate(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::Consolidate)?;
        let from = self.parse_ident()?;
        self.expect(&TokenKind::Arrow)?;
        let to = self.parse_ident()?;
        let threshold = if self.eat_keyword(Keyword::Threshold) {
            self.expect(&TokenKind::Assign)?;
            Some(self.parse_expr(0)?)
        } else {
            None
        };
        let span = match &threshold {
            Some(value) => keyword.span.merge(value.span()),
            None => keyword.span.merge(to.span),
        };
        Ok(Item::Consolidate {
            from,
            to,
            threshold,
            span,
        })
    }

    fn parse_recall(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::Recall)?;
        let memory = self.parse_ident()?;
        let arguments = self.parse_eq_fields(true)?;
        let span = keyword.span.merge(self.previous_span());
        Ok(Item::Recall {
            memory,
            arguments,
            span,
        })
    }

    fn parse_plan(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::Plan)?;
        let name = self.parse_ident()?;
        let body = self.parse_colon_fields()?;
        let span = keyword.span.merge(self.previous_span());
        Ok(Item::Plan { name, body, span })
    }

    fn parse_self_model(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::SelfModel)?;
        let name = self.parse_ident()?;
        let body = self.parse_colon_fields()?;
        let span = keyword.span.merge(self.previous_span());
        Ok(Item::SelfModel { name, body, span })
    }

    fn parse_architecture(&mut self) -> Result<ArchitectureBlock, ParseError> {
        let keyword = self.expect_keyword(Keyword::Architecture)?;
        let name = self.parse_ident()?;
        self.expect(&TokenKind::LBrace)?;
        let mut operations = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            operations.push(self.parse_arch_op()?);
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        let close = self.expect(&TokenKind::RBrace)?;
        let span = keyword.span.merge(close.span);
        Ok(ArchitectureBlock {
            name,
            operations,
            span,
        })
    }

    fn parse_arch_op(&mut self) -> Result<ArchOp, ParseError> {
        let token = self.peek().clone();
        match token.kind {
            TokenKind::Keyword(Keyword::Add) => {
                self.bump();
                self.expect_keyword(Keyword::Component)?;
                let component = self.parse_expr(0)?;
                let span = token.span.merge(component.span());
                Ok(ArchOp::AddComponent { component, span })
            }
            TokenKind::Keyword(Keyword::Expand) => {
                self.bump();
                self.expect_keyword(Keyword::Memory)?;
                let memory = self.parse_expr(0)?;
                self.expect_keyword(Keyword::By)?;
                let by = self.parse_expr(0)?;
                let span = token.span.merge(by.span());
                Ok(ArchOp::ExpandMemory { memory, by, span })
            }
            TokenKind::Keyword(Keyword::Change) => {
                self.bump();
                self.expect_keyword(Keyword::Routing)?;
                let routing = self.parse_expr(0)?;
                let span = token.span.merge(routing.span());
                Ok(ArchOp::ChangeRouting { routing, span })
            }
            _ => Err(self.unexpected("`add`, `expand`, or `change`")),
        }
    }

    fn parse_society(&mut self) -> Result<SocietyBlock, ParseError> {
        let keyword = self.expect_keyword(Keyword::Society)?;
        let name = self.parse_ident()?;
        self.expect(&TokenKind::LBrace)?;
        let mut members = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            members.push(self.parse_society_member()?);
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        let close = self.expect(&TokenKind::RBrace)?;
        let span = keyword.span.merge(close.span);
        Ok(SocietyBlock {
            name,
            members,
            span,
        })
    }

    fn parse_society_member(&mut self) -> Result<SocietyMember, ParseError> {
        let token = self.peek().clone();
        match token.kind {
            TokenKind::Keyword(Keyword::Agent) => {
                self.bump();
                let name = self.parse_ident()?;
                let fields = self.parse_colon_fields()?;
                let span = token.span.merge(self.previous_span());
                Ok(SocietyMember::Agent { name, fields, span })
            }
            TokenKind::Keyword(Keyword::Cluster) => {
                self.bump();
                let name = self.parse_ident()?;
                self.expect(&TokenKind::Assign)?;
                self.expect(&TokenKind::LBracket)?;
                let mut members = Vec::new();
                if !self.at(&TokenKind::RBracket) {
                    loop {
                        members.push(self.parse_ident()?);
                        if !self.eat(&TokenKind::Comma) {
                            break;
                        }
                    }
                }
                let close = self.expect(&TokenKind::RBracket)?;
                let span = token.span.merge(close.span);
                Ok(SocietyMember::Cluster {
                    name,
                    members,
                    span,
                })
            }
            _ => Err(self.unexpected("`agent` or `cluster`")),
        }
    }

    fn parse_route(&mut self) -> Result<Item, ParseError> {
        let keyword = self.expect_keyword(Keyword::Route)?;
        let request = self.parse_ident()?;
        self.expect_keyword(Keyword::To)?;
        let cluster = self.parse_ident()?;
        self.expect_keyword(Keyword::Budget)?;
        let budget = self.parse_colon_fields()?;
        let span = keyword.span.merge(self.previous_span());
        Ok(Item::Route {
            request,
            cluster,
            budget,
            span,
        })
    }

    // ── types ───────────────────────────────────────────────────────

    /// Parses `f32`, `i64`, `f32[2, 3]`, or shape-generic `f32[N]`.
    pub fn parse_type(&mut self) -> Result<TypeExpr, ParseError> {
        let name = self.parse_ident()?;
        let mut span = name.span;
        let mut dims = Vec::new();
        if self.eat(&TokenKind::LBracket) {
            loop {
                let token = self.peek().clone();
                let dim = match token.kind {
                    TokenKind::Integer(value) => {
                        self.bump();
                        TypeDim::Const {
                            value: value.max(0) as u64,
                            span: token.span,
                        }
                    }
                    TokenKind::Ident(_) | TokenKind::Keyword(_) => {
                        TypeDim::Dynamic(self.parse_ident()?)
                    }
                    _ => return Err(self.unexpected("a tensor dimension")),
                };
                span = span.merge(dim.span());
                dims.push(dim);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            let close = self.expect(&TokenKind::RBracket)?;
            span = span.merge(close.span);
        }
        Ok(TypeExpr::new(name, dims, span))
    }

    // ── shared helpers ──────────────────────────────────────────────

    /// Parses a name position, accepting reserved words as plain names so
    /// that body fields such as `horizon: 10` or `because: "..."` work.
    pub fn parse_ident(&mut self) -> Result<Ident, ParseError> {
        let token = self.peek().clone();
        match token.kind {
            TokenKind::Ident(name) => {
                self.bump();
                Ok(Ident::new(name, token.span))
            }
            TokenKind::Keyword(keyword) => {
                self.bump();
                Ok(Ident::new(keyword.as_str(), token.span))
            }
            _ => Err(self.unexpected("an identifier")),
        }
    }

    /// Parses `{ name: value, ... }` (trailing comma allowed).
    pub fn parse_colon_fields(&mut self) -> Result<Vec<Field>, ParseError> {
        self.expect(&TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            let name = self.parse_ident()?;
            self.expect(&TokenKind::Colon)?;
            let value = self.parse_expr(0)?;
            let span = name.span.merge(value.span());
            fields.push(Field::new(name, value, span));
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RBrace)?;
        Ok(fields)
    }

    /// Parses whitespace-separated `name = value` pairs.
    pub fn parse_eq_fields(&mut self, require_one: bool) -> Result<Vec<Field>, ParseError> {
        let mut fields = Vec::new();
        while self.at_ident_like() && self.peek_second_is(&TokenKind::Assign) {
            let name = self.parse_ident()?;
            self.expect(&TokenKind::Assign)?;
            let value = self.parse_expr(0)?;
            let span = name.span.merge(value.span());
            fields.push(Field::new(name, value, span));
        }
        if require_one && fields.is_empty() {
            return Err(self.unexpected("at least one `name = value` pair"));
        }
        Ok(fields)
    }

    // ── cursor primitives ───────────────────────────────────────────

    pub(crate) fn peek(&self) -> &Token {
        let index = self.pos.min(self.tokens.len().saturating_sub(1));
        &self.tokens[index]
    }

    pub(crate) fn peek_kind(&self) -> &TokenKind {
        &self.peek().kind
    }

    pub(crate) fn peek_second_is(&self, kind: &TokenKind) -> bool {
        self.tokens
            .get(self.pos + 1)
            .is_some_and(|token| &token.kind == kind)
    }

    pub(crate) fn bump(&mut self) -> Token {
        let token = self.peek().clone();
        if self.pos < self.tokens.len().saturating_sub(1) {
            self.pos += 1;
        }
        token
    }

    pub(crate) fn at(&self, kind: &TokenKind) -> bool {
        self.peek_kind() == kind
    }

    pub(crate) fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.at(kind) {
            self.bump();
            true
        } else {
            false
        }
    }

    pub(crate) fn at_keyword(&self, keyword: Keyword) -> bool {
        matches!(self.peek_kind(), TokenKind::Keyword(found) if *found == keyword)
    }

    pub(crate) fn eat_keyword(&mut self, keyword: Keyword) -> bool {
        if self.at_keyword(keyword) {
            self.bump();
            true
        } else {
            false
        }
    }

    pub(crate) fn at_ident_like(&self) -> bool {
        matches!(
            self.peek_kind(),
            TokenKind::Ident(_) | TokenKind::Keyword(_)
        )
    }

    pub(crate) fn expect(&mut self, kind: &TokenKind) -> Result<Token, ParseError> {
        if self.at(kind) {
            Ok(self.bump())
        } else {
            Err(self.unexpected(&format!("`{kind}`")))
        }
    }

    pub(crate) fn expect_keyword(&mut self, keyword: Keyword) -> Result<Token, ParseError> {
        if self.at_keyword(keyword) {
            Ok(self.bump())
        } else {
            Err(self.unexpected(&format!("`{keyword}`")))
        }
    }

    /// Span of the most recently consumed token.
    pub(crate) fn previous_span(&self) -> Span {
        let index = self.pos.saturating_sub(1);
        self.tokens
            .get(index)
            .map(|token| token.span)
            .unwrap_or_else(|| self.peek().span)
    }

    /// Builds an `expected ... found ...` error at the cursor.
    pub(crate) fn unexpected(&self, expected: &str) -> ParseError {
        let token = self.peek();
        ParseError::new(
            format!("expected {expected}, found {}", token.describe()),
            token.span,
        )
    }
}

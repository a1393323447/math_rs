use std::{collections::{VecDeque, HashMap}, cell::RefCell};

use crate::tokenizer::{ DFA, Token, ExprTokenType, get_expr_tokenizer};
use super::ast::{ASTNode, Opend, Operator, Variable, ResultType};

use lazy_static::*;

pub struct Expression {
    variable_table: HashMap<String, ASTNode>,
    main_expr: ASTNode,
}

impl Expression {
    pub fn parse(origin: &str) -> Self {
        parse(origin)
    }
    pub fn excute(&self) -> ResultType {
        self.main_expr.excute(&self.variable_table)
    }
    pub fn set_variable(&mut self, val_name: &str, val: &str) -> &mut Self {
        match self.variable_table.get_mut(val_name) {
            Some(expr) => {
                *expr = parse(val).main_expr;
            },
            None => { panic!("no variable name `{}` in expression", val_name) },
        }
        self
    }
}

struct Parser {
    tokenizer: DFA<ExprTokenType>,
    tokens: VecDeque<Token<ExprTokenType>>,
}

// 基础文法
// E -> num   |  var  |  (E)  | sin(E)| cos(E)| tan(E)| tanh(E) |
//      -E
//      E!    |
//      E * E | E / E | E % E |
//      E + E | E - E |

// 定义优先级
// (), sin, cos, tan, tanh 
// -
// !
// * , / , %
// + , -

// E -> E + T | E - T  | T
// T -> T * F | T / F  | T % F | F
// F -> G !   | G
// G -> - H   | H
// H -> (E)   | sin(E) | cos(E)| tan(E)
//            | tanh(E)| num   | var

// 消除左递归
// # 表示空结束符
// E  -> T E1
// E1 -> + T E1 | - T E1 | #
// T  -> F T1
// T1 -> * F T1 | / F T1 | % F T1 | #
// F -> G !   | G
// G -> - H   | H
// H -> (E)   | sin(E) | cos(E)| tan(E)
//            | tanh(E)| num   | var

impl Parser {
    fn new() -> Self {
        let tokenizer = get_expr_tokenizer();
        let tokens = VecDeque::new();
        Parser { tokenizer, tokens }
    }
    fn get_next_token(&mut self) -> Token<ExprTokenType> {
        self.tokens.pop_front().expect("[Syntax Error] unexpected ternimate.")
    }
    fn watch_next_token(&self) -> Option<&Token<ExprTokenType>> {
        self.tokens.front()
    }
    fn parse(&mut self, origin: &str) -> Expression {
        if !self.tokens.is_empty() {
            self.tokens.clear()
        }
        self.tokens = self.tokenizer.tokenize(origin);
        let mut variable_table = HashMap::new();
        self.tokens.iter().for_each(|t| {
            match t.ty {
                ExprTokenType::Var => { variable_table.insert(t.val.clone(), ASTNode::new(None, None, Operator::Add)); },
                _ => { /* Do nothing */ }
            }
        });
        let right_op = self.parse_e();
        let mut ast = ASTNode::new(Some(right_op), None, Operator::Add);
        while !self.tokens.is_empty() {
            let right_op = self.parse_e();
            ast.__set_right_op(right_op);
            ast = ASTNode::new(Some(Opend::Expr(Box::new(ast))), None, Operator::Add);
        }
        ast.__set_right_op(Opend::Num(0.0));
        Expression { variable_table, main_expr: ast }
    }
    fn parse_e(&mut self) -> Opend {
        let left_op = self.parse_t();
        let next_token = self.watch_next_token();
        match next_token {
            Some(token) => {
                match token.ty {
                    ExprTokenType::Add |
                    ExprTokenType::Sub => {
                        let right_half_expr = self.parse_e1();
                        let result = match right_half_expr {
                            Some(mut half_expr) => {
                                half_expr.__set_left_op(left_op);
                                let complete_expr = half_expr;
                                Opend::Expr(Box::new(complete_expr))
                            },
                            None => left_op,
                        };
                        result
                    },
                    _ => left_op,
                }
            },
            None => left_op,
        }
    }
    fn parse_e1(&mut self) -> Option<ASTNode> {
        let left_op = None;
        let next_token = self.watch_next_token();
        match next_token {
            Some(token) => {
                match token.ty {
                    ExprTokenType::Add => {
                        let _op = self.get_next_token();
                        let right_op = self.parse_t();
                        let ast_node = ASTNode::new(left_op, Some(right_op), Operator::Add);
                        Some(ast_node)
                    },
                    ExprTokenType::Sub => {
                        let _op = self.get_next_token();
                        let right_op = self.parse_t();
                        let ast_node = ASTNode::new(left_op, Some(right_op), Operator::Sub);
                        Some(ast_node)
                    },
                    _ => { panic!("[Syntax Error] Unexpected token {:?}", token) }
                }
            },
            None => None,
        }
    }
    fn parse_t(&mut self) -> Opend {
        let left_op = self.parse_f();
        let next_token = self.watch_next_token();
        match next_token {
            Some(token) => {
                match token.ty {
                    ExprTokenType::Mul |
                    ExprTokenType::Div |
                    ExprTokenType::Mod => {
                        let right_half_expr = self.parse_t1();
                        let result = match right_half_expr {
                            Some(mut half_expr) => {
                                half_expr.__set_left_op(left_op);
                                let complete_expr = half_expr;
                                Opend::Expr(Box::new(complete_expr))
                            },
                            None => left_op,
                        };
                        result
                    },
                    _ => left_op,
                }
            },
            None => left_op,
        }
    }
    fn parse_t1(&mut self) -> Option<ASTNode> {
        let left_op = None;
        let next_token = self.watch_next_token();
        match next_token {
            Some(token) => {
                match token.ty {
                    ExprTokenType::Mul => {
                        let _op = self.get_next_token();
                        let right_op = self.parse_f();
                        let ast_node = ASTNode::new(left_op, Some(right_op), Operator::Mul);
                        Some(ast_node)
                    },
                    ExprTokenType::Div => {
                        let _op = self.get_next_token();
                        let right_op = self.parse_f();
                        let ast_node = ASTNode::new(left_op, Some(right_op), Operator::Div);
                        Some(ast_node)
                    },
                    ExprTokenType::Mod => {
                        let _op = self.get_next_token();
                        let right_op = self.parse_f();
                        let ast_node = ASTNode::new(left_op, Some(right_op), Operator::Mod);
                        Some(ast_node)
                    },
                    _ => { panic!("[Syntax Error] Unexpected token {:?}", token) }
                }
            },
            None => None,
        }
    }
    fn parse_f(&mut self) -> Opend {
        let left_op = self.parse_g();
        let next_token = self.watch_next_token();
        match next_token {
            Some(t) => {
                match t.ty {
                    ExprTokenType::Fac => {
                        let _fac = self.get_next_token();
                        let operator = Operator::Fac;
                        let right_op = Opend::Num(0.0);
                        let expr = ASTNode::new(Some(left_op), Some(right_op), operator);
                        Opend::Expr(Box::new(expr))
                    },
                    _ => { left_op }
                }
            },
            None => { left_op },
        }
    }
    fn parse_g(&mut self) -> Opend {
        let next_token = self.watch_next_token();
        match next_token {
            Some(t) => {
                match t.ty {
                    ExprTokenType::Sub => {
                        let _sub = self.get_next_token();
                        let left_op = Opend::Num(0.0);
                        let operator = Operator::Sub;
                        let right_op = self.parse_h();
                        let expr = ASTNode::new(Some(left_op), Some(right_op), operator);
                        Opend::Expr(Box::new(expr))
                    },
                    _ => { self.parse_h() }
                }
            },
            None => { panic!("[Syntax Error] unexpected ternimate.") },
        }
    }
    fn parse_h(&mut self) -> Opend {
        let next_token = self.tokens.pop_front().expect("unexpected ternimate.");

        let result = match next_token.ty {
            ExprTokenType::LeftScope => {
                let expr = self.parse_e();
                let _right_scope = self.get_next_token();
                expr
            }
            ExprTokenType::Var => {
                let var = Variable::new(next_token.val);
                Opend::Var(var)
            },
            ExprTokenType::Num => {
                let num = next_token.val.parse().expect("Parse num error.");
                Opend::Num(num)
            },
            ExprTokenType::Sin => {
                self.parse_func(Operator::Sin)
            },
            ExprTokenType::Cos => {
                self.parse_func(Operator::Cos)
            },
            ExprTokenType::Tan => {
                self.parse_func(Operator::Tan)
            },
            ExprTokenType::Tanh => {
                self.parse_func(Operator::Tanh)
            },
            _ => { panic!("[Syntax Error] Unexpeted token {:?}", next_token) },
        };
        result
    }
    fn parse_func(&mut self, operator: Operator) -> Opend {
        let left_scope = self.watch_next_token();
        match left_scope {
            Some(_) => {
                let token = self.get_next_token();
                match token.ty {
                    ExprTokenType::LeftScope => {
                        let left_op = self.parse_e();
                        let right_op = Opend::Num(0.0);
                        let func_expr = ASTNode::new(Some(left_op), Some(right_op), operator);
                        let right_scope = self.watch_next_token();
                        match right_scope {
                            Some(token) => {
                                match token.ty {
                                    ExprTokenType::RigthScope => { self.get_next_token(); }
                                    _ => { panic!("[Syntax Error] Unexpected token {:?}", token) }
                                }
                            }
                            None => { panic!("[Syntax Error] Expr `func(..)` missing rightscope ')' .") },
                        }
                        Opend::Expr(Box::new(func_expr))
                    },
                    _ => { panic!("[Syntax Error] Unexpected token {:?}", token) },
                }
            },
            None => { panic!("[Syntax Error] Expr `func(..)` missing leftscope '(' .") },
        }
    }
}

struct ExprParser {
    inner: RefCell<Parser>,
}

unsafe impl Send for ExprParser {}
unsafe impl Sync for ExprParser {}

lazy_static! {
    static ref EXPR_PARSE: ExprParser = {
        let parser = Parser::new();
        let inner = RefCell::new(parser);
        ExprParser { inner }
    };
}

fn parse(origin: &str) -> Expression {
    EXPR_PARSE.inner.borrow_mut().parse(origin)
}

#[test]
fn test_parse() {
    let mut expr = Expression::parse("-x1 + x2 * 3 - sin(3.14) + (cos(3.14) - 2 * cos(3.14))");
    expr
        .set_variable("x1", "2 * 231")
        .set_variable("x2", "2");
    let pi: f64 = 3.14;
    println!("{}", -2.0 * 231.0 + 2.0 * 3.0 - pi.sin() + (pi.cos() - pi.cos() * 2.0));
    println!("{}", expr.excute());
}
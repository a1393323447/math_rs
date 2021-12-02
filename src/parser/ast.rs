use std::collections::HashMap;

pub type ResultType = f64;

#[derive(Debug)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    Fac,
    Mod,
    Sin,
    Cos,
    Tan,
    Tanh,
}

#[derive(Debug)]
pub struct Variable {
    symbol: String,
}

impl Variable {
    pub fn new(symbol: String) -> Self {
        Self { symbol }
    }
    fn get(&self, variable_table: &HashMap<String, ASTNode>) -> ResultType {
        match variable_table.get(&self.symbol) {
            Some(expr) => expr.excute(variable_table),
            None => panic!("variable `{}` has no value.", self.symbol),
        }
    }
}

#[derive(Debug)]
pub enum Opend {
    Num(ResultType),
    Expr(Box<ASTNode>),
    Var(Variable),
}

impl Opend {
    pub fn get(&self, variable_table: &HashMap<String, ASTNode>) -> ResultType {
        match self {
            Opend::Num(num) => *num,
            Opend::Expr(expr) => expr.excute(variable_table),
            Opend::Var(v) => v.get(variable_table),
        }
    }
}

#[derive(Debug)]
pub struct ASTNode {
    left_op: Option<Opend>,
    right_op: Option<Opend>,
    operator: Operator,
}

impl ASTNode {
    pub fn new(left_op: Option<Opend>, right_op: Option<Opend>, operator: Operator) -> Self {
        ASTNode { left_op, right_op, operator }
    }
    pub fn excute(&self, variable_table: &HashMap<String, ASTNode>) -> ResultType {
        let left = self.left_op.as_ref().unwrap().get(variable_table);
        let right = self.right_op.as_ref().unwrap().get(variable_table);
        match self.operator {
            Operator::Add => { left + right },
            Operator::Sub => { left - right },
            Operator::Mul => { left * right },
            Operator::Div => { 
                if right.eq(&0.0) { panic!("div by 0."); } 
                else { left / right } 
            },
            Operator::Mod => {
                if right.eq(&0.0) { panic!("div by 0."); } 
                else { left % right } 
            },
            Operator::Fac => { todo!() },
            Operator::Sin => { left.sin() },
            Operator::Cos => { left.cos() },
            Operator::Tan => { left.tan() },
            Operator::Tanh => { left.tanh() },
        }
    }
    pub fn __set_left_op(&mut self, left_op: Opend) {
        self.left_op = Some(left_op)
    }
    pub fn __set_right_op(&mut self, right_op: Opend) {
        self.right_op = Some(right_op)
    }
}

#[test]
fn test_excute() {
    let op1 = Opend::Num(1.0);
    let op2 = Opend::Num(1.0);
    let ast = ASTNode::new(Some(op1), Some(op2), Operator::Add);
    let op1 = Opend::Num(2.0);
    let op2 = Opend::Expr(Box::new(ast));
    let ast = ASTNode::new(Some(op1), Some(op2), Operator::Mul);
    let op1 = Opend::Num(3.14);
    let op2 = Opend::Num(0.0);
    let sin_ast = ASTNode::new(Some(op1), Some(op2), Operator::Sin);
    let op1 = Opend::Expr(Box::new(ast));
    let op2 = Opend::Expr(Box::new(sin_ast));
    let final_ast = ASTNode::new(Some(op1), Some(op2), Operator::Add);
    let table = HashMap::new();
    println!("{}", final_ast.excute(&table));
}
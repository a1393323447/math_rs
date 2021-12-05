use super::dfa::*;
use super::nfa::*;
use super::token::ExprTokenType;
use crate::link_nfa;

pub fn get_expr_tokenizer() -> DFA<ExprTokenType> {

    let digit = NFA::from_symbol_range('0'..='9');
    let no_zero = NFA::from_symbol_range('1'..='9');
    let non_zero_int = no_zero.clone() & digit.clone().closure();
    let int = non_zero_int.clone() | NFA::from_symbol('0');
    let mut num = int.clone() & 
                                   (NFA::zero_or_one(".") |
                                    NFA::zero_or_one("e") | 
                                    NFA::zero_or_one("e-")
                                   ) & 
                                   int.clone().closure();
    num.set_state(ExprTokenType::Num);

    let all_alp = NFA::from_symbol_set("xyzknmpqijXYZKNMPQIJ");
    let mut var = all_alp & digit.clone().closure();
    var.set_state(ExprTokenType::Var);

    let mut add = NFA::from_symbol('+');
    add.set_state(ExprTokenType::Add);

    let mut sub = NFA::from_symbol('-');
    sub.set_state(ExprTokenType::Sub);

    let mut mul = NFA::from_symbol('*');
    mul.set_state(ExprTokenType::Mul);

    let mut fac = NFA::from_symbol('!');
    fac.set_state(ExprTokenType::Fac);

    let mut mod_ = NFA::from_symbol('%');
    mod_.set_state(ExprTokenType::Mod);

    let mut div = NFA::from_symbol('/');
    div.set_state(ExprTokenType::Div);

    let mut sin = NFA::from_literal("sin");
    sin.set_state(ExprTokenType::Sin);

    let mut cos = NFA::from_literal("cos");
    cos.set_state(ExprTokenType::Cos);

    let mut tan = NFA::from_literal("tan");
    tan.set_state(ExprTokenType::Tan);

    let mut tanh = NFA::from_literal("tanh");
    tanh.set_state(ExprTokenType::Tanh);

    let mut left_scope = NFA::from_symbol('(');
    left_scope.set_state(ExprTokenType::LeftScope);

    let mut right_scope = NFA::from_symbol(')');
    right_scope.set_state(ExprTokenType::RigthScope);

    let mut blank = NFA::from_symbol(' ');
    blank.set_state(ExprTokenType::Blank);

    let tokenizer_nfa = link_nfa!(var, num, add, sub, mul, div, fac, mod_, sin, cos, tan, tanh, left_scope, right_scope, blank);

    DFA::subset_construct(&tokenizer_nfa)
}

#[test]
fn test_tokenizer() {
    let tokenizer = get_expr_tokenizer();
    // let result = tokenizer.match_one("tanh");
    // println!("{:?}", result);
    let origin = "x1+x2+x3*4 + tanh(1.3) + sin(3.2) + cos(1.0) - tan(1/2.0) + 1 % 2 + 31e-22";
    let result = tokenizer.tokenize(origin);
    println!("{:?}", result);
}
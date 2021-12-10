pub trait TokenType: Copy {
    fn is_nonterminal(&self) -> bool;
}

#[derive(Clone, Copy, Debug)]
pub enum ExprTokenType {
    Var,
    Num,
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
    LeftScope,
    RigthScope,
    Blank,
}

impl TokenType for ExprTokenType {
    fn is_nonterminal(&self) -> bool {
        match self {
            ExprTokenType::Blank => true,
            _ => false,
        }
    }
}

#[derive(Debug)]
pub struct Token<TokenType> {
    pub val: String,
    pub ty: TokenType,
}

impl<TokenType> Token<TokenType>  {
    pub fn new(val: String, ty: TokenType) -> Token<TokenType> {
        Token { val, ty }
    }
}
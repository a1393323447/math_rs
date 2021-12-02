mod tokenizer;
mod parser;

pub use crate::parser::parser::Expression;

#[cfg(test)]
mod tests {
    use crate::Expression;

    #[test]
    fn it_works() {
        let mut expr = Expression::parse("1 + 2*3 - x1 * x2");
        expr.set_variable("x1", "2")
            .set_variable("x2", "2 * 2");
        let result = expr.excute();
        println!("result = {}", result);
        expr.set_variable("x1", "sin(3.14)")
            .set_variable("x2", "1");
        let result = expr.excute();
        println!("result = {}", result);
    }
}

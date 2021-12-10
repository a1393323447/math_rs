//! Big Num \
//! This crate provides:
//! - [`BigInt`]: Immutable arbitrary-precision integers.  All operations behave as if BigInt were represented in two's-complement notation.
//! - `BigDec`: Immutable, arbitrary-precision signed decimal numbers. A BigDecimal consists of an arbitrary precision integer unscaled value and a 32-bit integer scale. (Coming Soon)

mod big_int;
mod big_num_cache;
mod big_num_constants;

pub use big_int::BigInt;

#[cfg(test)]
mod tests {
    use crate::BigInt;

    #[test]
    fn it_works() {
        let a: BigInt = "10000000000000".into();
        let b: BigInt = "900000000000".into();
        println!("a = {}", a);      
        println!("a + b = {}", &a + &b);
        println!("a - b = {}", &a - &b);
        println!("a * b = {}", &a * &b);
        println!("a / b = {}", &a / &b);
        println!("a % b = {}", &a % &b);
        println!("a << 10 = {}", &a << 10);
        println!("a >> 10 = {}", &a >> 10);
    }
}

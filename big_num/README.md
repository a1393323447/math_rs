# Big Num
This crate provides:
- `BigInt`: Immutable arbitrary-precision integers.  All operations behave as if BigInt were represented in two's-complement notation.
- `BigDec`: Immutable, arbitrary-precision signed decimal numbers. A BigDecimal consists of an arbitrary precision integer unscaled value and a 32-bit integer scale. (Coming Soon)
# Example
```rust
use big_num::BigInt;

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
```
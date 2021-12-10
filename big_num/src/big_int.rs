//! # BigInt
//! Immutable arbitrary-precision integers.  All operations behave as if BigInt were represented in two's-complement notation.
//! Range form `-2^(u32::MAX)` to `2^(u32::MAX)`.
//! # Example
//! ```
//! use big_num::BigInt;
//!
//! let a: BigInt = "10000000000000".into();
//! let b: BigInt = "900000000000".into();
//! println!("a = {}", a);      
//! println!("a + b = {}", &a + &b);
//! println!("a - b = {}", &a - &b);
//! println!("a * b = {}", &a * &b);
//! println!("a / b = {}", &a / &b);
//! println!("a % b = {}", &a % &b);
//! println!("a << 10 = {}", &a << 10);
//! println!("a >> 10 = {}", &a >> 10);
//! ```
//! 

use std::fmt::Display;
use std::ops::{
    Add, AddAssign,
    Sub, SubAssign,
    Mul, MulAssign,
    Div, DivAssign,
    Shl, ShlAssign,
    Shr, ShrAssign,
    Rem, RemAssign,
    Neg, 
};
use std::cmp::{Ord, Eq, PartialEq, PartialOrd, Ordering};

use crate::big_num_constants::*;
use crate::big_num_cache::*;

pub const ZERO: BigInt = BigInt { signum: 0, mag: vec![], bit_len_plus_one: 0 };

macro_rules! new_zero_vec_with_cap {
    ($cap: expr) => {
        {
            let mut v = Vec::with_capacity($cap as usize);
            unsafe { v.set_len($cap as usize); }
            v.iter_mut().for_each(|n| *n = 0u32 );
            v
        }
    };
}

macro_rules! skip_leading_zero {
    ($vec: expr) => {
        {
            $vec
                .into_iter()
                .skip_while(|x| *x == 0)
                .collect()
        }
    };
}

macro_rules! bit_length_u32 {
    ($n: expr) => {
        (32 - $n.leading_zeros()) as usize
    };
}

#[derive(Debug, Clone)]
pub struct BigInt {
    signum: i8,
    mag: Vec<u32>,
    bit_len_plus_one: i32,
}

// 杂项辅助函数
impl BigInt {
    /// 计算数组的前 len 元素的比特数，假设没有前导 0
    fn bit_length(val: &Vec<u32>, len: usize) -> usize {
        if len == 0 {
            0
        } else {
            ((len - 1) << 5) + bit_length_u32!(val[0])
        }
    }
    fn self_bit_length(&self) -> i32 {
        let mut n = self.bit_len_plus_one - 1;
        if n == -1 { // bitLength not initialized yet
            let len = self.mag.len();
            if len == 0 {
                n = 0; // offset by one to initialize
            } else {
                // Calculate the bit length of the magnitude
                let mag_bit_len = ((len - 1) << 5) + bit_length_u32!(self.mag[0]);
                let mag_bit_len = mag_bit_len as i32;
                if self.signum < 0 {
                    // Check if magnitude is a power of two
                    let mut is_pow2 = self.mag[0].count_ones() == 1;
                    for i in 1..len {
                        if !is_pow2 { break; }
                        is_pow2 = self.mag[i] == 0;
                    }
                    n = if is_pow2 { mag_bit_len - 1 } else { mag_bit_len };
                } else {
                    n = mag_bit_len;
                }
            }
            unsafe {
                let addr = std::ptr::addr_of!(self.bit_len_plus_one) as usize;
                let ptr = addr as *mut i32;
                *ptr = n + 1; 
            }
        }
        n
    }
}

// 实现构造
impl BigInt {
    pub unsafe fn from_raw(mag: Vec<u32>, signum: i8) -> Self {
        BigInt::new(mag, signum)
    }
    fn new(mag: Vec<u32>, signum: i8) -> Self {
        BigInt { signum, mag, bit_len_plus_one: 0 }
    }
}

// 实现打印
impl Display for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string_radix(10))
    }
}

impl BigInt {
    pub fn to_string_radix(&self, mut radix: u32) -> String {
        if self.signum == 0 {
            return String::from("0");
        }
        if radix < 2 || radix > 36 {
            radix = 10;
        }

        // If it's small enough, use smallToString.
        if self.mag.len() <= SCHOENHAGE_BASE_CONVERSION_THRESHOLD {
            return self.small_to_string(radix);
        }

        // Otherwise use recursive toString, which requires positive arguments.
        // The results will be concatenated into this String.
        let mut s: String = String::new();
        if self.signum < 0 {
            let u = self.clone().neg();
            BigInt::to_string_recursive(&u, &mut s, radix, 0);
            s.insert(0, '-');
        } else {
            BigInt::to_string_recursive(self, &mut s, radix, 0);
        }
        
        s
    }
    fn small_to_string(&self, radix: u32) -> String {
        if self.signum == 0 {
            return String::from("0");
        }
        // Compute upper bound on number of digit groups and allocate space
        let max_num_digit_groups = (4 * self.mag.len() + 6) / 7;
        let mut digit_group: Vec<String> = Vec::with_capacity(max_num_digit_groups);

        // Translate number to string, a digit group at a time
        let mut tmp = self.abs();
        let mut num_groups = 0;
        while tmp.signum != 0 {
            let d = LONG_RADIX[radix as usize].clone();
            let r = tmp.clone() % d.clone();
            let mut val = r.to_u64();
            // val to string
            let mut val_s = String::new();
            while val != 0 {
                let r = val % radix as u64;
                val_s.push(DIGITS[r as usize]);
                val /= radix as u64;
            }
            digit_group.push(val_s.chars().rev().collect());
            tmp /= d;
            num_groups += 1;
        }
        let mut result = String::with_capacity(num_groups * DIGITS_PER_LONG[radix as usize]);
        if self.signum < 0 {
            result.push('-');
        }
        result.extend(digit_group[num_groups - 1].chars());

        // Append remaining digit groups padded with leading zeros
        for i in (0..num_groups-1).rev() {
            // Prepend (any) leading zeros for this digit group
            let num_leading_zero = DIGITS_PER_LONG[radix as usize] as isize - digit_group[i].len() as isize;
            for _ in 0..num_leading_zero {
                result.push('0');
            }
            result.extend(digit_group[i].chars());
        }
        result
    }
    /// Converts the specified BigInteger to a string and appends to s. 
    /// This implements the recursive Schoenhage algorithm for base conversions.
    /// See Knuth, Donald, _The Art of Computer Programming_, Vol. 2, Answers to Exercises (4.4) Question 14.
    fn to_string_recursive(u: &BigInt, s: &mut String, radix: u32, digits: i32) {
        // If we're smaller than a certain threshold, use the smallToString
        // method, padding with leading zeroes when necessary.
        if u.mag.len() <= SCHOENHAGE_BASE_CONVERSION_THRESHOLD {
            let u_s = u.small_to_string(radix);

            // Pad with internal zeros if necessary.
            // Don't pad if we're at the beginning of the string.
            if u_s.len() < digits as usize && s.len() > 0 {
                for _ in s.len()..digits as usize {
                    s.push('0');
                }
            }

            s.extend(u_s.chars());
            return;
        }
        
        let b = u.self_bit_length();
        
        // Calculate a value for n in the equation radix^(2^n) = u
        // and subtract 1 from that value.  This is used to find the
        // cache index that contains the best value to divide u.
        let n = (((b as f64 * LOG_CACHE[2]) / LOG_CACHE[radix as usize]).ln() / LOG_CACHE[2] - 1.0) as i32;
        // radix^(2^exponent)
        let mut v = BigInt::from(radix);
        for _ in 0..n {
            v *= v.clone();
        }
        let q = u.clone() / v.clone();
        let r = u.clone() % v;

        let expected_digits = 1 << n;

        BigInt::to_string_recursive(&q, s, radix, digits - expected_digits);
        BigInt::to_string_recursive(&r, s, radix, expected_digits);
    }
}

// 实现解析
impl From<&str> for BigInt {
    fn from(val: &str) -> Self {
        BigInt::from_str_radix(val, 10)
    }
}

macro_rules! impl_unsigned_to_big_num {
    ($($u: ty),*) => {
    $(
    impl From<$u> for BigInt {
        fn from(val: $u) -> Self {
            if val == 0 {
                BigInt::value_of(val as u64, 0)
            } else {
                BigInt::value_of(val as u64, 1)
            }
        }
    }
    )*
    };
}

macro_rules! impl_signed_to_big_num {
    ($($i: ty),*) => {
    $(
    impl From<$i> for BigInt {
        fn from(val: $i) -> Self {
            if val == 0 {
                BigInt::value_of(val as u64, 0)
            } else if val < 0 {
                BigInt::value_of((-val) as u64, -1)
            } else {
                BigInt::value_of(val as u64, 1)
            }
        }
    }
    )*
    };
}
impl_unsigned_to_big_num!(u8, u16, u32, usize, u64);
impl_signed_to_big_num!(i8, i16, i32, isize, i64);

impl BigInt {
    fn value_of(val: u64, signum: i8) -> BigInt {
        if val == 0 {
            return ZERO;
        } else if  val <= MAX_CONSTANT as u64 {
            if signum == 1 {
                return POS_CACHE[val as usize].clone();
            } else {
                return NEG_CACHE[val as usize].clone();
            }
        } else {
            let high = val >> 32;
            let mag = if high == 0 {
                vec![val as u32]
            } else {
                vec![high as u32, val as u32]
            };
            return BigInt::new(mag, signum);
        }
    }
    fn to_u64(self) -> u64 {
        if self.signum == 0 || self.mag.len() == 0 {
            0
        } else if self.mag.len() == 1 {
            self.mag[0] as u64
        } else {
            ((self.mag[0] as u64) << 32) + self.mag[1] as u64
        }
    }
    pub fn from_str_radix(val: &str, radix: u32) -> BigInt {
        let mut cursor: usize = 0;
        let num_digits: usize;
        let len = val.len();

        if radix < 2 && radix > 36 {
            panic!("Radix out bound Error");
        }

        if len == 0 {
            panic!("Zero length BigInt");
        }

        let mut signum: i8 = 1;
        let index_plus = val.rfind('+');
        let index_mis = val.rfind('-');
        match (index_plus, index_mis) {
            (Some(_), Some(_)) => panic!("Multiple sign charater"),
            (Some(plus_pos), None) => { 
                if plus_pos > 0 {
                    panic!("Illegal embedded sign charater") 
                } else {
                    cursor = 1;
                }
            },
            (None, Some(min_pos)) => { 
                if min_pos > 0 { 
                    panic!("Illegal embedded sign charater") 
                } else {
                    signum = -1;
                    cursor = 1;
                }
            },
            (None, None) => { /* Do nothing */ },
        }

        if cursor == len {
            // TODO
            panic!("Zero length BigInt");
        }
        
        // skip leading zero
        let iter = val.chars().skip(cursor);
        for c in iter {
            match c.to_digit(10) {
                Some(0) => { cursor += 1; },
                _ => { break; },
            }
        }

        if cursor == len {
            return ZERO.clone();
        }

        num_digits = len - cursor;
        let num_words = if len < 10 {
            1
        } else {
            let num_bits = ((num_digits * unsafe { *BITS_PER_DIGIT.get_unchecked(10) }) >> 10) + 1;
            if num_bits >= usize::MAX - 31 {
                panic!("over flow");
            }
            (num_bits + 31) >> 5
        };

        let mut magnitude: Vec<u32> = new_zero_vec_with_cap!(num_words);

        let mut first_group_len = num_digits % DIGITS_PER_INT[radix as usize];
        if first_group_len == 0 {
            first_group_len = DIGITS_PER_INT[radix as usize];
        }
        let mut group = &val[cursor..cursor + first_group_len];
        cursor += first_group_len;
        magnitude[num_words - 1] = u32::from_str_radix(group, radix).expect("Todo");

        let super_radix = INT_RADIX[radix as usize];
        let mut group_val;
        while cursor < len {
            group = &val[cursor..cursor+DIGITS_PER_INT[radix as usize]];
            group_val = u32::from_str_radix(group, radix).expect("TODO");
            BigInt::destructive_mul_add(&mut magnitude, super_radix, group_val);
            cursor += DIGITS_PER_INT[radix as usize];
        }
        let mag = skip_leading_zero!(magnitude);
        BigInt::new(mag, signum)
    }

    // 以 2^32 为 radix 编码 BigInt
    #[inline(always)]
    fn destructive_mul_add(magnitude: &mut Vec<u32>, super_radix: u32, group_val: u32) {
        let mut product: u64;
        let mut carry = 0;
        for x in magnitude.iter_mut().rev() {
            product = (super_radix as u64) * (*x as u64) + carry;
            *x = product as u32;
            carry = product >> u32::BITS;
        }

        let mut sum = *magnitude.last().unwrap() as u64 + group_val as u64;
        *magnitude.last_mut().unwrap() = sum as u32;
        carry = sum as u64 >> u32::BITS;

        for x in magnitude.iter_mut().rev().skip(1) {
            sum = *x as u64 + carry;
            *x = sum as u32;
            carry = (sum >> u32::BITS) as u64;
        }
    }
}

// 实现大小比较
impl BigInt {
    fn compare_mag(&self, other: &BigInt) -> std::cmp::Ordering {
        let self_mag = &self.mag;
        let other_mag = &other.mag;
        let self_len = self_mag.len();
        let other_len = other_mag.len();

        if self_len < other_len {
            return std::cmp::Ordering::Less;
        } 
        
        if self_len > other_len {
            return std::cmp::Ordering::Greater;
        }

        let mut pos = 0;

        while pos < self_len {
            let a = unsafe { *self_mag.get_unchecked(pos)  };
            let b = unsafe { *other_mag.get_unchecked(pos) };
            if a != b {
                return a.cmp(&b);
            }
            pos += 1;
        }

        return std::cmp::Ordering::Equal;
    }
}
impl PartialEq for BigInt {
    fn eq(&self, other: &Self) -> bool {
        self.signum == other.signum && self.compare_mag(&other).is_eq()
    }
}
impl Eq for BigInt {}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.signum.partial_cmp(&other.signum) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        if self.signum > 0 {
            Some(self.compare_mag(&other))
        } else {
            Some(self.compare_mag(&other).reverse())
        }
    }
}

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(&other).unwrap()
    }
}

// 实现绝对值
impl BigInt {
    pub fn abs(&self) -> BigInt {
        self.clone().abs_take()
    }
    fn abs_take(self) -> BigInt {
        let BigInt { signum, mag, bit_len_plus_one } = self;
        let signum = signum.abs();
        BigInt { signum, mag, bit_len_plus_one }
    }
}

// 实现加法
impl Add for BigInt {
    type Output = BigInt;

    fn add(self, val: Self) -> Self::Output {
        if val.signum == 0 {
            return self;
        }

        if self.signum == 0{
            return val;
        }

        if val.signum == self.signum {
            let signum = self.signum;
            return BigInt::new(BigInt::add(self.mag, val.mag), signum);
        }

        match self.compare_mag(&val) {
            Ordering::Less => {
                let signum = -self.signum;
                let mag = BigInt::sub(val.mag, self.mag);
                let mag = skip_leading_zero!(mag);
                BigInt::new(mag, signum)               
            },
            Ordering::Equal => ZERO,
            Ordering::Greater => {
                let signum = self.signum;
                let mag = BigInt::sub(self.mag, val.mag);
                let mag = skip_leading_zero!(mag);
                BigInt::new(mag, signum)
            },
        }
    }
}

impl BigInt {
    fn add(mut x: Vec<u32>, mut y: Vec<u32>) -> Vec<u32> {
        if x.len() < y.len() {
            std::mem::swap(&mut x, &mut y);
        }

        let mut x_index = x.len();
        let mut y_index = y.len();
        let mut result = new_zero_vec_with_cap!(x_index);
        let mut sum: u64 = 0;
        if y_index == 1 {
            x_index -= 1;
            sum = x[x_index] as u64 + y[0] as u64;
            result[x_index] = sum as u32;
        } else {
            while y_index > 0 {
                x_index -= 1;
                y_index -= 1;
                sum = x[x_index] as u64 +
                      y[y_index] as u64 +
                      (sum >> u32::BITS);
                result[x_index] = sum as u32;
            }
        }

        let mut carry = (sum >> 32) != 0;
        while x_index > 0 && carry {
            x_index -= 1;
            let x_val = x[x_index] + 1;
            result[x_index] = x_val;
            carry = x_val == 0;
        }

        while x_index > 0 {
            x_index -= 1;
            let x_val = x[x_index];
            result[x_index] = x_val;
        }

        if carry {
            result.insert(0, 0x01);
        }

        result
    }
}

impl AddAssign for BigInt {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl Add for &BigInt {
    type Output = BigInt;

    fn add(self, rhs: Self) -> Self::Output {
        self.clone() + rhs.clone()
    }
}

impl AddAssign<&BigInt> for BigInt {
    fn add_assign(&mut self, rhs: &BigInt) {
        *self = self.clone() + rhs.clone();
    }
}

// 实现取反
impl Neg for BigInt {
    type Output = BigInt;

    fn neg(self) -> Self::Output {
        let BigInt { signum: sign, mag, bit_len_plus_one } = self;
        BigInt { signum: -sign, mag, bit_len_plus_one }
    }
}

impl Neg for &BigInt {
    type Output = BigInt;

    fn neg(self) -> Self::Output {
        self.clone().neg()
    }
}

// 实现减法
impl Sub for BigInt {
    type Output = BigInt;

    fn sub(self, val: Self) -> Self::Output {
        if val.signum == 0 {
            return self;
        }

        if self.signum == 0{
            return -val;
        }

        if val.signum != self.signum {
            return BigInt::new(BigInt::add(self.mag, val.mag), self.signum)
        }

        match self.compare_mag(&val) {
            Ordering::Less => {
                let signum = -self.signum;
                let mag = BigInt::sub(val.mag, self.mag);
                let mag = skip_leading_zero!(mag);
                BigInt::new(mag, signum)               
            },
            Ordering::Equal => ZERO,
            Ordering::Greater => {
                let signum = self.signum;
                let mag = BigInt::sub(self.mag, val.mag);
                let mag = skip_leading_zero!(mag);
                BigInt::new(mag, signum)
            },
        }
    }
}

impl BigInt {
    fn sub(big: Vec<u32>, little: Vec<u32>) -> Vec<u32> {
        let mut big_index = big.len();
        let mut little_index = little.len();
        let mut difference: i64 = 0;
        let mut result = new_zero_vec_with_cap!(big_index);

        while little_index > 0 {
            big_index -= 1;
            little_index -= 1;
            difference = big[big_index] as i64 -
                         little[little_index] as i64 +
                         (difference >> u32::BITS);
            result[big_index] = difference as u32;
        }

        let mut borrow = (difference as i64) >> 32 != 0;
        while big_index > 0 && borrow {
            big_index -= 1;
            let val = big[big_index] - 1;
            result[big_index] = val;
            borrow = val == (-1_i32 as u32);
        }

        while big_index > 0 {
            big_index -= 1;
            let val = big[big_index]; 
            result[big_index] = val; 
        }
        
        result
    }
}

impl SubAssign for BigInt {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl Sub for &BigInt {
    type Output = BigInt;

    fn sub(self, rhs: Self) -> Self::Output {
        self.clone() - rhs.clone()
    }
}

impl SubAssign<&BigInt> for BigInt {
    fn sub_assign(&mut self, rhs: &BigInt) {
        *self = self.clone() - rhs.clone();
    }
}

// 实现左移
impl Shl<u32> for BigInt {
    type Output = BigInt;

    fn shl(self, n: u32) -> Self::Output {
        let signum = self.signum;
        if signum == 0 {
            ZERO
        } else if n == 0 {
            self
        } else {
            BigInt::new(BigInt::shl(self.mag, n), signum)
        }
    }
}

impl BigInt {
    fn shl(mag: Vec<u32>, n: u32) -> Vec<u32> {
        let n_ints = n >> 5;
        let n_bits = n & 0x1f; // n % 32
        let mag_len = mag.len();
        let new_mag = {
            if n_bits == 0 {
                let mut new_m = new_zero_vec_with_cap!(mag_len + n_ints as usize);
                unsafe {
                    std::ptr::copy(mag.as_ptr(), new_m.as_mut_ptr(), mag_len);
                }
                new_m
            } else {
                let mut i = 0;
                let n_bits_2 = 32 - n_bits;
                let high_bits = mag[0] >> n_bits_2;
                let mut new_m = if high_bits != 0 {
                    let mut new_m = new_zero_vec_with_cap!(mag_len + n_ints as usize + 1);
                    new_m[0] = high_bits;
                    i += 1;
                    new_m
                } else {
                    new_zero_vec_with_cap!(mag_len + n_ints as usize)
                };
                if mag_len + i > new_m.len() {
                    panic!("Out of bound");
                }
                let num_iter = mag_len as isize - 1;
                assert!(num_iter >= 0);
                BigInt::shl_impl(&mut new_m, &mag, i, n_bits, num_iter as usize);
                let val = mag[num_iter as usize];
                new_m[num_iter as usize + i] = val << n_bits;
                new_m
            }
        };
        new_mag
    }

    #[inline(always)]
    fn shl_impl(new_vec: &mut Vec<u32>, old_vec: &Vec<u32>, mut new_index: usize, shift_cnt: u32, num_iter: usize) {
        let shift_cnt_r = u32::BITS - shift_cnt;
        for i in 0..num_iter {
            unsafe {
                let new_high = *old_vec.get_unchecked(i) << shift_cnt;
                let new_low = *old_vec.get_unchecked(i + 1) >> shift_cnt_r;
                *new_vec.get_unchecked_mut(new_index) = new_high | new_low;
                new_index += 1;
            }
        }
    }
}

impl Shl<u32> for &BigInt {
    type Output = BigInt;

    fn shl(self, rhs: u32) -> Self::Output {
        self.clone() << rhs
    }
}

impl ShlAssign<u32> for BigInt {
    fn shl_assign(&mut self, n: u32) {
        *self = self.clone() << n;
    }
}

// 实现右移
impl Shr<u32> for BigInt {
    type Output = BigInt;

    fn shr(self, n: u32) -> Self::Output {
        let signum = self.signum;
        
        if signum == 0 {
            ZERO
        } else if n == 0 {
            self
        } else {
            BigInt::shr(self.mag, n, signum)
        }
    }
}

impl BigInt {
    fn shr(mag: Vec<u32>, n: u32, signum: i8) -> BigInt {
        let n_ints = n >> 5;
        let n_bits = n & 0x1f; // n % 32
        let mag_len = mag.len();

        if n_ints as usize >= mag_len {
            if signum >= 0 {
                return ZERO;
            } else {
                return NEG_CACHE[1].clone();
            }
        }

        let mut new_mag = {
            if n_bits == 0 {
                let new_mag_len = mag_len - n_ints as usize;
                mag[0..new_mag_len].to_vec()
            } else {
                let mut i = 0;
                let high_bits = mag[0] >> n_bits;
                let mut new_m = if high_bits != 0 {
                    let mut new_m = new_zero_vec_with_cap!(mag_len - n_ints as usize);
                    new_m[0] = high_bits;
                    i += 1;
                    new_m
                } else {
                    new_zero_vec_with_cap!(mag_len - n_ints as usize - 1)
                };
                let num_iter = mag_len as isize - n_ints as isize - 1;
                assert!(num_iter >= 0);
                let num_iter = num_iter as usize;
                if num_iter + i > new_m.len() {
                    panic!("Out of bound");
                }
                BigInt::shr_impl(&mut new_m, &mag, i, n_bits, num_iter);
                new_m
            }
        };

        if signum < 0 {
            let mut one_lost = false;
            let mut i = mag_len - 1;
            let j = mag_len - n_ints as usize;
            
            while i >= j && !one_lost {
                one_lost = mag[i] != 0;
                i -= 1;
            }

            if !one_lost && n_bits != 0 {
                one_lost = 
                ((mag[mag_len - n_ints as usize - 1] as u64 ) << (32 - n_bits as u64)) != 0;
            }

            if one_lost {
                let new_m_len = new_mag.len();
                let mut last_sum: i32 = 0;
                for val in new_mag.iter_mut().rev() {
                    *val = *val + 1;
                    last_sum = *val as i32;
                    if last_sum != 0 {
                        break;
                    }
                }
                if last_sum == 0 {
                    new_mag = new_zero_vec_with_cap!(new_m_len + 1);
                    new_mag[0] = 1;
                }
            }
        }

        BigInt::new(new_mag, signum)
    }

    #[inline(always)]
    fn shr_impl(new_vec: &mut Vec<u32>, old_vec: &Vec<u32>, new_index: usize, shift_cnt: u32, num_iter: usize) {
        let shift_cnt_l = u32::BITS - shift_cnt;
        let mut old_index = num_iter;
        let mut n_index = if new_index == 0 { num_iter - 1 } else { num_iter };
        while n_index as isize >= new_index as isize {
            unsafe {
                let new_low = *old_vec.get_unchecked(old_index) >> shift_cnt;
                old_index -= 1;
                let new_high = *old_vec.get_unchecked(old_index) << shift_cnt_l;
                *new_vec.get_unchecked_mut(n_index) = new_high | new_low;
                n_index = (n_index as isize - 1) as usize;
            }
        }
    }
}

impl Shr<u32> for &BigInt {
    type Output = BigInt;

    fn shr(self, rhs: u32) -> Self::Output {
        self.clone() << rhs
    }
}

impl ShrAssign<u32> for BigInt {
    fn shr_assign(&mut self, n: u32) {
        *self = self.clone() >> n; 
    }
}

// 实现乘法
impl Mul for BigInt {
    type Output = BigInt;

    fn mul(self, val: Self) -> Self::Output {
        BigInt::mul(self, val, false)
    }
}

impl BigInt {
    fn mul(self, val: BigInt, is_recursion: bool) -> BigInt {
        let self_signum = self.signum;
        let val_signum = val.signum;
        if self_signum == 0 || val_signum == 0 {
            return ZERO;
        }

        let x_len = self.mag.len();

        if self == val && x_len > MULTIPLY_SQUARE_THRESHOLD {
            return self.square();
        }

        let y_len = val.mag.len();

        if x_len < KARATSUBA_THRESHOLD || y_len < KARATSUBA_THRESHOLD {
            let result_signum = if self_signum == val_signum { 1 } else { -1 };
            if y_len == 1 {
                return BigInt::mul_by_int(self.mag, val.mag[0], result_signum);
            }
            if x_len == 1 {
                return BigInt::mul_by_int(val.mag, self.mag[0], result_signum);
            }
            let result = BigInt::mul_to_len(self.mag, x_len, val.mag, y_len, None);
            return BigInt::new(skip_leading_zero!(result), result_signum);
        } else if x_len < TOOM_COOK_THRESHOLD &&
                  y_len < TOOM_COOK_THRESHOLD {
            return BigInt::mul_karatsuba(self, val);
        } else {
            //
            // In "Hacker's Delight" section 2-13, p.33, it is explained
            // that if x and y are unsigned 32-bit quantities and m and n
            // are their respective numbers of leading zeros within 32 bits,
            // then the number of leading zeros within their product as a
            // 64-bit unsigned quantity is either m + n or m + n + 1. If
            // their product is not to overflow, it cannot exceed 32 bits,
            // and so the number of leading zeros of the product within 64
            // bits must be at least 32, i.e., the leftmost set bit is at
            // zero-relative position 31 or less.
            //
            // From the above there are three cases:
            //
            //     m + n    leftmost set bit    condition
            //     -----    ----------------    ---------
            //     >= 32    x <= 64 - 32 = 32   no overflow
            //     == 31    x >= 64 - 32 = 32   possible overflow
            //     <= 30    x >= 64 - 31 = 33   definite overflow
            //
            // The "possible overflow" condition cannot be detected by
            // examning data lengths alone and requires further calculation.
            //
            // By analogy, if 'this' and 'val' have m and n as their
            // respective numbers of leading zeros within 32*MAX_MAG_LENGTH
            // bits, then:
            //
            //     m + n >= 32*MAX_MAG_LENGTH        no overflow
            //     m + n == 32*MAX_MAG_LENGTH - 1    possible overflow
            //     m + n <= 32*MAX_MAG_LENGTH - 2    definite overflow
            //
            // Note however that if the number of ints in the result
            // were to be MAX_MAG_LENGTH and mag[0] < 0, then there would
            // be overflow. As a result the leftmost bit (of mag[0]) cannot
            // be used and the constraints must be adjusted by one bit to:
            //
            //     m + n >  32*MAX_MAG_LENGTH        no overflow
            //     m + n == 32*MAX_MAG_LENGTH        possible overflow
            //     m + n <  32*MAX_MAG_LENGTH        definite overflow
            //
            // The foregoing leading zero-based discussion is for clarity
            // only. The actual calculations use the estimated bit length
            // of the product as this is more natural to the internal
            // array representation of the magnitude which has no leading
            // zero elements.
            //
            if !is_recursion {
                if BigInt::bit_length(&self.mag, x_len) +
                   BigInt::bit_length(&val.mag, y_len)  >
                   32 * MAX_MAG_LENGTH {
                    panic!("Overflow!");
                }
            }
            return BigInt::mul_toom_cook3(self, val);
        }
    }
    pub fn square(self) -> BigInt {
        self.square_impl(false)
    }
    fn square_impl(self, is_recursion: bool) -> BigInt {
        if self.signum == 0 {
            return ZERO;
        }
        let len = self.mag.len();

        if len < KARATSUBA_SQUARE_THRESHOLD {
            let z = BigInt::square_to_len(self.mag, len, None);
            return BigInt::new(skip_leading_zero!(z), 1);
        } else if len < TOOM_COOK_SQUARE_THRESHOLD {
            return self.square_karatsuba();
        } else {
            if !is_recursion {
                if BigInt::bit_length(&self.mag, self.mag.len()) > (16_usize * MAX_MAG_LENGTH) {
                    panic!("Overflow");
                }
            }
            return self.square_toom_cook3();
        }
    }
    fn square_to_len(x: Vec<u32>, len: usize, mut z: Option<Vec<u32>>) -> Vec<u32> {
        let z_len = len << 1;
        if z.is_none() || z.as_ref().unwrap().len() < z_len {
            z = Some(new_zero_vec_with_cap!(z_len));
        }
        let vec_x_len = x.len();
        let vec_z_len = z.as_ref().unwrap().len();
        // check
        if len < 1 {
            panic!("invalid input length: {}", len);
        }
        if len > vec_x_len {
            panic!("input length out of bound: {} > {}", len, vec_x_len);
        }
        if len * 2 > vec_z_len {
            panic!("input length out of bound: {} > {}", len * 2, vec_z_len);
        }
        if z_len < 1 {
            panic!("invalid input length: {}", z_len);
        }
        if z_len > vec_z_len {
            panic!("input length out of bound: {} > {}", len, vec_z_len);
        }

        /*
         * The algorithm used here is adapted from Colin Plumb's C library.
         * Technique: Consider the partial products in the multiplication
         * of "abcde" by itself:
         *
         *               a  b  c  d  e
         *            *  a  b  c  d  e
         *          ==================
         *              ae be ce de ee
         *           ad bd cd dd de
         *        ac bc cc cd ce
         *     ab bb bc bd be
         *  aa ab ac ad ae
         *
         * Note that everything above the main diagonal:
         *              ae be ce de = (abcd) * e
         *           ad bd cd       = (abc) * d
         *        ac bc             = (ab) * c
         *     ab                   = (a) * b
         *
         * is a copy of everything below the main diagonal:
         *                       de
         *                 cd ce
         *           bc bd be
         *     ab ac ad ae
         *
         * Thus, the sum is 2 * (off the diagonal) + diagonal.
         *
         * This is accumulated beginning with the diagonal (which
         * consist of the squares of the digits of the input), which is then
         * divided by two, the off-diagonal added, and multiplied by two
         * again.  The low bit is simply a copy of the low bit of the
         * input, so it doesn't need special care.
         */

        // Store the squares, right shifted one bit (i.e., divided by 2)
        let mut z = z.unwrap();

        let mut last_product_low_word = 0;
        let mut z_index: usize = 0;
        for x_index in 0..len {
            let piece: u64 = x[x_index] as u64;
            let product = piece * piece;
            z[z_index] = (last_product_low_word << 31) | ((product >> 33) as u32);
            z_index += 1;
            z[z_index] = (product >> 1) as u32;
            z_index += 1;
            last_product_low_word = product as u32;
        }

        let mut offset = 1;
        for i in (1..=len).rev() {
            let mut t = x[i - 1];
            t = BigInt::mul_add(&mut z, &x, offset, i - 1, t as u64);
            BigInt::add_one(&mut z, offset - 1, i, t);
            offset += 2;
        }
        // Shift back up and set low bit
        BigInt::primitive_left_shift(&mut z, z_len, 1);
        let val = x[len - 1] & 1;
        z[z_len - 1] |= val;
        
        z
    }
    fn mul_add(out_v: &mut Vec<u32>, in_v: &Vec<u32>, 
               offset: isize, len: usize, k: u64) -> u32 {
        // check
        let out_len = out_v.len();
        let in_len = in_v.len();
        if len > in_len {
            panic!("input length is out of bound: {} > {}", len, in_len);
        }
        if offset < 0 {
            panic!("input offset is invalid: {}", offset);
        }
        if offset > (out_len as isize - 1) {
            panic!("input offset is out of bound: {} > {}", offset, out_len as isize - 1);
        }
        if (len as isize) > (out_len as isize - offset) {
            panic!("input len is out of bound: {} > {}", len, out_len as isize - offset as isize);
        }

        let mut carry: u64 = 0;

        let mut offset = out_len - (offset as usize) - 1;
        let iter_times = len as isize - 1;
        if iter_times < 0 {
            return carry as u32;
        }
        let iter_times = iter_times as usize;
        for in_index in (0..=iter_times).rev() {
            let product: u64 = ((in_v[in_index]   as u64) * k) +
                                 (out_v[offset]   as u64)      +
                                 carry;
            out_v[offset] = product as u32;
            offset -= 1;
            carry = product >> 32;
        }
        return carry as u32;
    }

    fn add_one(a: &mut Vec<u32>, offset: isize, mlen: usize, carry: u32) -> u32 {
        let offset = a.len() - 1 - mlen - offset as usize;
        let t = (a[offset] as u64) + (carry as u64);

        a[offset] = t as u32;
        if (t >> 32) == 0 {
            return 0;
        }
        let mut offset = offset as isize;
        for _ in 0..mlen {
            offset -= 1;
            if offset < 0 {
                return 1;
            } else {
                a[offset as usize] += 1;
                if a[offset as usize] != 0 {
                    return 0;
                }
            }
        }
        return 1;
    }

    /// shifts a up to len left n bits assumes no leading zeros, 0<=n<32
    fn primitive_left_shift(a: &mut Vec<u32>, len: usize, n: u32) {
        if len == 0 || n == 0 {
            return;
        }
        if len > a.len() {
            panic!("Out of bound");
        }
        let b = a.clone();
        BigInt::shl_impl(a, &b, 0, n, len - 1);
        a[len - 1] <<= n;
    }
    fn square_karatsuba(self) -> BigInt {
        let half = (self.mag.len() + 1) / 2;

        let x_low  = BigInt::get_lower(self.mag.clone(), self.signum, half);
        let x_high = BigInt::get_upper(self.mag,half);
        let xhs = x_high.clone().square();
        let xls = x_low.clone().square();
        let xlhs = (x_low.clone() + x_high.clone()).square();
        
        // xh^2 << 64  +  (((xl+xh)^2 - (xh^2 + xl^2)) << 32) + xl^2
        let shift_len = half as u32 * 32;
        (    (   (xhs.clone() << shift_len)  +   xlhs   -   (xhs + xls.clone()   )   ) << shift_len) + xls
    }
    /// Returns a new BigInteger representing n lower ints of the number. 
    /// This is used by Karatsuba multiplication and Karatsuba squaring.
    fn get_lower(mag: Vec<u32>, signum: i8, n: usize) -> BigInt {
        let len = mag.len();

        if len <= n {
            return BigInt::new(mag, signum.abs());
        }

        let mut lower_ints = new_zero_vec_with_cap!(n);
        unsafe {
            std::ptr::copy(mag.as_ptr().add(len - n), lower_ints.as_mut_ptr(), n);
        }

        let result = BigInt::new(skip_leading_zero!(lower_ints), 1);

        if result.mag.len() == 0 {
            ZERO
        } else {
            result
        }
    }
    // Returns a new BigInteger representing mag.length-n upper ints of the number. 
    // This is used by Karatsuba multiplication and Karatsuba squaring.
    fn get_upper(mag: Vec<u32>, n: usize) -> BigInt {
        let len = mag.len();

        if len <= n {
            return ZERO;
        }

        let upper_len = len - n;
        let mut upper_ints = new_zero_vec_with_cap!(upper_len);
        unsafe {
            std::ptr::copy(mag.as_ptr(), upper_ints.as_mut_ptr(), upper_len);
        }

        let result = BigInt::new(skip_leading_zero!(upper_ints), 1);
        if result.mag.len() == 0 {
            ZERO
        } else {
            result
        }
    }
    fn square_toom_cook3(self) -> BigInt {
        let len = self.mag.len();

        // k is the size (in ints) of the lower-order slices.
        let k = (len + 2) / 3;

        // r is the size (in ints) of the highest-order slice.
        let r = len - 2 * k;

        // Obtain slices of the numbers. a2 is the most significant
        // bits of the number, and a0 the least significant.
        let a2 = self.get_toom_slice(k, r, 0, len);
        let a1 = self.get_toom_slice(k, r, 1, len);
        let a0 = self.get_toom_slice(k, r, 2, len);
        
        let v0 = a0.clone().square_impl(true);
        let da1 = a2.clone() + a0.clone();
        let c = da1.clone() - a1.clone();
        let vm1 = c.square_impl(true);
        let da1 = da1 + a1.clone();
        let v1 = da1.clone().square();
        let vinf = a2.clone().square();
        let v2 = (((da1 + a2) << 1) - a0).square_impl(true);

        // The algorithm requires two divisions by 2 and one by 3.
        // All divisions are known to be exact, that is, they do not produce
        // remainders, and all results are positive.  The divisions by 2 are
        // implemented as right shifts which are relatively efficient, leaving
        // only a division by 3.
        // The division by 3 is done by an optimized algorithm for this case.
        let t2 = (v2 - vm1.clone()).exact_divide_by_3();
        let tm1 = (v1.clone() - vm1.clone()) >> 1;
        let t1 = v1 - v0.clone();
        let t2 = (t2 - t1.clone()) >> 1;
        let t1 = t1 - tm1.clone() - vinf.clone();
        let t2 = t2 - (vinf.clone() << 1);
        let tm1 = tm1 - t2.clone();

        // Number of bits to shift left.
        let ss = (k * 32) as u32;

        let result = (vinf   << ss) + t2;
        let result = (result << ss) + t1;
        let result = (result << ss) + tm1;
        let result = (result << ss) + v0;

        result
    }
    fn get_toom_slice(&self, low_size: usize, upper_size: usize, 
                      slice: usize, fullsize: usize) -> BigInt {
        let len = self.mag.len();
        let offset: isize = fullsize as isize - len as isize;

        let (start, end) = if slice == 0 {
            (0 - offset, upper_size as isize - 1 - offset)
        } else {
            let start = upper_size as isize + (slice as isize - 1) * low_size as isize - offset;
            let end = start + low_size as isize - 1;
            (start, end)
        };

        if end < 0 {
            return ZERO;
        }

        let start = if start < 0 { 0 } else { start };

        let slice_size = (end - start) + 1;

        if slice_size < 0 {
            return ZERO;
        }

        let start = start as usize;
        let slice_size = slice_size as usize;

        if start == 0 && slice_size >= len {
            return self.abs();
        }

        let mut int_slice = new_zero_vec_with_cap!(slice_size);
        unsafe {
            std::ptr::copy(self.mag.as_ptr().add(start), int_slice.as_mut_ptr(), slice_size);
        }

        BigInt::new(skip_leading_zero!(int_slice), 1)
    }
    fn exact_divide_by_3(&self) -> BigInt {
        let len = self.mag.len();
        let mut result = new_zero_vec_with_cap!(len);
        let mut borrow: u64 = 0;
        for i in (0..=len-1).rev() {
            let x = self.mag[i] as u64;
            let w = x - borrow;
            if borrow > x {
                borrow = 1;
            } else {
                borrow = 0;
            }
            // 0xAAAAAAAB is the modular inverse of 3 (mod 2^32).  Thus,
            // the effect of this is to divide by 3 (mod 2^32).
            // This is much faster than division on most architectures.
            let q = (w * 0xAAAAAAAB) as u32;
            result[i] = q;
            // Now check the borrow. The second check can of course be
            // eliminated if the first fails.
            if q >= 0x55555556 {
                borrow += 1;
                if q >= 0xAAAAAAAB {
                    borrow += 1;
                }
            }
        }
        BigInt::new(skip_leading_zero!(result), self.signum)
    }
    fn mul_by_int(x: Vec<u32>, y: u32, signum: i8) -> BigInt {
        if y.count_ones() == 1 {
            return BigInt::new(BigInt::shl(x, y.trailing_zeros()), signum);
        }
        let x_len = x.len();
        let result_len = x_len + 1;
        let mut result_mag = new_zero_vec_with_cap!(result_len);
        let mut carry: u64 = 0;
        let mut result_start = x_len;
        for x_index in (0..=x_len-1).rev() {
            let product: u64 = (x[x_index] as u64) * (y as u64) + carry;
            result_mag[result_start] = product as u32;
            result_start -= 1;
            carry = product >> 32;
        }
        if carry == 0 {
            result_mag = result_mag[1..].to_vec();
        } else {
            result_mag[result_start] = carry as u32;
        }
        
        BigInt::new(result_mag, signum)
    }

    fn mul_to_len(x: Vec<u32>, x_len: usize, y: Vec<u32>, y_len: usize, mut z: Option<Vec<u32>>) -> Vec<u32> {
        // here: x_len > 0, y_len > 0
        let x_start = x_len - 1;
        let y_start = y_len - 1;

        let z_len_at_least = x_len + y_len;

        if z.is_none() || z.as_ref().unwrap().len() < z_len_at_least {
            z = Some(new_zero_vec_with_cap!(z_len_at_least));
        }

        let mut carry = 0;

        let mut z = z.unwrap();

        let mut z_index = y_start + 1 + x_start;
        for y_index in (0..=y_start).rev() {
            let product: u64 = (y[y_index] as u64) *
                                 (x[x_start] as u64) +
                                 carry;
            z[z_index] = product as u32;
            z_index -= 1;
            carry = product >> 32; 
        }
        z[x_start] = carry as u32;

        for x_index in (0..=x_start-1).rev() {
            carry = 0;
            let mut z_index = y_start + 1 + x_index;
            for y_index in (0..=y_start).rev() {
                let product: u64 = (y[y_index] as u64) *
                                     (x[x_index] as u64) +
                                     (z[z_index] as u64) +
                                     carry;
                z[z_index] = product as u32;
                z_index -= 1;
                carry = product >> 32;
            }
            z[x_index] = carry as u32;
        }

        z
    }
    /// Multiplies two BigIntegers using the Karatsuba multiplication
    /// algorithm.  This is a recursive divide-and-conquer algorithm which is
    /// more efficient for large numbers than what is commonly called the
    /// "grade-school" algorithm used in multiplyToLen.  If the numbers to be
    /// multiplied have length n, the "grade-school" algorithm has an
    /// asymptotic complexity of `O(n^2)`.  In contrast, the Karatsuba algorithm
    /// has complexity of `O(n^(log2(3)))`, or `O(n^1.585)`.  It achieves this
    /// increased performance by doing 3 multiplies instead of 4 when
    /// evaluating the product.  As it has some overhead, should be used when
    /// both numbers are larger than a certain threshold (found
    /// experimentally).
    /// See:  http://en.wikipedia.org/wiki/Karatsuba_algorithm
    fn mul_karatsuba(x: BigInt, y: BigInt) -> BigInt {
        let x_len = x.mag.len();
        let y_len = y.mag.len();

        // The number of ints in each half of the number.
        let half = (x_len.max(y_len) + 1) / 2;

        // xl and yl are the lower halves of x and y respectively,
        // xh and yh are the upper halves.
        let xl = BigInt::get_lower(x.mag.clone(), x.signum, half);
        let xh = BigInt::get_upper(x.mag, half);
        let yl = BigInt::get_lower(y.mag.clone(), y.signum, half);
        let yh = BigInt::get_upper(y.mag, half);

        let p1 = xh.clone() * yh.clone();
        let p2 = xl.clone() * yl.clone();
        let p3 = (xh + xl) * (yh + yl);
        let p4 = p3 - p1.clone() - p2.clone();
        let shift_len = (32 * half) as u32;
        // // result = p1 * 2^(32*2*half) + (p3 - p1 - p2) * 2^(32*half) + p2
        let result = (((p1 << shift_len) + p4) << shift_len) + p2;

        if x.signum != y.signum {
            -result
        } else {
            result
        }
    }
    fn mul_toom_cook3(a: BigInt, b: BigInt) -> BigInt {
        let a_len = a.mag.len();
        let b_len = b.mag.len();

        let largest = a_len.max(b_len);
        // // k is the size (in ints) of the lower-order slices.
        let k = (largest + 2) / 3;

        // r is the size (in ints) of the highest-order slice.
        let r = largest - 2 * k;

        // Obtain slices of the numbers. a2 and b2 are the most significant
        // bits of the numbers a and b, and a0 and b0 the least significant.
        let a2 = a.get_toom_slice(k, r, 0, largest);
        let a1 = a.get_toom_slice(k, r, 1, largest);
        let a0 = a.get_toom_slice(k, r, 2, largest);
        let b2 = b.get_toom_slice(k, r, 0, largest);
        let b1 = b.get_toom_slice(k, r, 1, largest);
        let b0 = b.get_toom_slice(k, r, 2, largest);

        // v0 = a0 * b0
        let v0  = a0.clone().mul(b0.clone(), true);
        // da1 = a2 + a0
        let da1 = a2.clone() + a0.clone();
        // db1 = b2 + b0
        let db1 = b2.clone() + b0.clone();
        // vm1 = (da1 - a1) * (db1 - b1)
        let vm1 = (da1.clone() - a1.clone())
                        .mul(db1.clone() - b1.clone(), true);
        let da1 = da1 + a1;
        let db1 = db1 + b1;
        // v1 = da1 * db1
        let v1 = da1.clone().mul(db1.clone(), true);
        // v2 = (((da1 + a2) << 1) - a0) * (((db1 + b2) << 1) - b0)
        let v2 = (((da1 + a2.clone()) << 1) - a0).mul(
                     ((db1 + b2.clone()) << 1) - b0
        , true);
        // vinf = a2 * b2
        let vinf = a2.mul(b2, true);
        
        // The algorithm requires two divisions by 2 and one by 3.
        // All divisions are known to be exact, that is, they do not produce
        // remainders, and all results are positive.  The divisions by 2 are
        // implemented as right shifts which are relatively efficient, leaving
        // only an exact division by 3, which is done by a specialized
        // linear-time algorithm.
        let t2  = (v2 - vm1.clone()).exact_divide_by_3();
        let tm1 = (v1.clone() - vm1) >> 1;
        let t1  = v1 - v0.clone();
        let t2  = (t2 - t1.clone()) >> 1;
        let t1  = t1 - tm1.clone() - vinf.clone();
        let t2  = t2 - (vinf.clone() << 1);
        let tm1 = tm1 - t2.clone();
        
        // Number of bits to shift left.
        let ss = (k * 32) as u32;

        let result = (vinf   << ss) + t2;
        let result = (result << ss) + t1;
        let result = (result << ss) + tm1;
        let result = (result << ss) + v0;
        
        if a.signum != b.signum {
            -result
        } else {
            result
        }
        
    }
}

impl Mul<&BigInt> for &BigInt {
    type Output = BigInt;

    fn mul(self, rhs: &BigInt) -> Self::Output {
        self.clone() * rhs.clone()
    }
}

impl MulAssign for BigInt {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl MulAssign<&BigInt> for BigInt {
    fn mul_assign(&mut self, rhs: &BigInt) {
        *self = self.clone() * rhs.clone()
    }
}

impl Div for BigInt {
    type Output = BigInt;

    fn div(self, rhs: Self) -> Self::Output {
        let w_len = self.mag.len();
        let w_signum = self.signum;
        let v_len = rhs.mag.len();
        let v_signum = rhs.signum;

        if self < rhs || w_signum == 0 {
            return ZERO;
        } else if v_signum == 0 {
            panic!("Divide by 0");
        } else if w_len == 1 && v_len == 1 {
            let w = self.mag[0];
            let v = rhs.mag[0];
            let q = w / v;
            let signum = w_signum * v_signum;
            return BigInt::new(vec![q], signum);
        } else if w_len == 1 || v_len == 1 {
            return BigInt::divide_one_word(self, rhs);
        } else {
            return BigInt::divide_knuth(self, rhs);
        }
    }
}

impl BigInt {
    /// Knuth "The Art Of Computer Programming" Vol.2 section 4.3.1 exercise 16
    fn divide_one_word(mut num_u: BigInt, num_v: BigInt) -> BigInt {
        let u = &mut num_u.mag;
        let v = num_v.mag[0];

        let mut r = 0;
        let n = u.len();
        let mut w = new_zero_vec_with_cap!(n);

        for j in 0..n {
            let sum: u64 = (r << 32) + u[j] as u64;
            w[j] = (sum / (v as u64)) as u32;
            r = sum % (v as u64);
        }

        let signum = num_u.signum * num_v.signum;

        BigInt::new(skip_leading_zero!(w), signum)
    }
    /// Uses Algorithm D in Knuth "The Art Of Computer Programming" Vol.2 section 4.3.1.
    fn divide_knuth(mut u: BigInt, mut v: BigInt) -> BigInt {
        let u_len = u.mag.len(); // m + n
        let v_len = v.mag.len(); // n
        let n = v_len;
        let m = u_len - v_len;

        // D1 normalize
        // Vn-1 * d >= b / 2
        // here b = 2^32
        // and  Vn-1 is u32
        // bits of Vn-1: [0..xxxx]
        // shift Vn-1 to [xxx0..0]
        // so Vn-1 >= [01000..0](2^31)
        let shift_len = v.mag[0].leading_zeros();
        if shift_len == 0 {
            u.mag.push(0);
        } else {
            u = u << shift_len;
            v = v << shift_len;
        }
        let m = if u.mag.len() <= n + m {
            m - 1
        } else {
            m
        };
        // D2
        // let j = m
        let b: u64  = 1 << 32;
        let mut q = new_zero_vec_with_cap!(m + 1);
        let u_len   = u.mag.len();
        for j in (0..=m).rev() {
            let sum = ((u.mag[u_len - 1 - n - j]   as u64) << 32) +
                            u.mag[u_len - n - j] as u64;
            // D3 calculate qhat
            let mut qhat: u64 = sum / (v.mag[0] as u64);
            let mut r: u64    = sum % (v.mag[0] as u64);
            // check
            while (r >> 32_u64) == 0 {
                if qhat != b {
                    let product = qhat * v.mag[1] as u64;
                    let sum = (r << 32_u64) + (u.mag[u_len + 1 - j - n] as u64);
                    if product > sum {
                        qhat -= 1;
                        r += v.mag[0] as u64;
                    } else {
                        break;
                    }
                } else {
                    qhat -= 1;
                    r += v.mag[0] as u64;
                }
            }
            let result = BigInt::mul_sub(&mut u.mag, &v.mag, qhat, n, j);
            q[m - j] = qhat as u32;
            if result < 0 {
                q[m - j] -= 1;
                BigInt::add_back(&mut u.mag, &v.mag, n, j);
            }
        }
        let signum = u.signum * v.signum;
        BigInt::new(skip_leading_zero!(q), signum)
    }
    fn mul_sub(u: &mut Vec<u32>, v: &Vec<u32>, qhat: u64, n: usize, j: usize) -> i8 {
        let mut a_mag = new_zero_vec_with_cap!(n + 1);
        let offset = u.len() - 1 - j - n;
        unsafe {
            std::ptr::copy(u.as_ptr().add(offset), a_mag.as_mut_ptr(), n+1);
        }
        let a = BigInt::new(skip_leading_zero!(a_mag), 1);

        let mut b_mag = new_zero_vec_with_cap!(n);
        let offset = v.len() - n;
        unsafe {
            std::ptr::copy(v.as_ptr().add(offset), b_mag.as_mut_ptr(), n);
        }
        let b = BigInt::new(b_mag, 1);
        
        let high = (qhat >> 32) as u32;
        let c = if high == 0 {
            BigInt::new(vec![qhat as u32], 1)
        } else {
            BigInt::new(vec![high, qhat as u32], 1)
        };
        let result = a -  c * b;
        let BigInt { signum, mag, .. } = result;
        let result_mag = if mag.len() < n + 1 {
            let mut new = new_zero_vec_with_cap!(n+1);
            let offset = n+1 - mag.len();
            unsafe {
                std::ptr::copy(mag.as_ptr(), new.as_mut_ptr().add(offset), mag.len());
            }
            new
        } else {
            mag
        };
        unsafe {
            std::ptr::copy(result_mag.as_ptr(), u.as_mut_ptr().add(u.len() - j - n - 1), n + 1);
        }
        signum
    }
    fn add_back(u: &mut Vec<u32>, v: &Vec<u32>, n: usize, j: usize) {
        let start = v.len() - n;
        let a_mag = v[start..].to_vec();
        let a = BigInt::new(a_mag, 1);

        let start = u.len() - 1 - j - n;
        let end = u.len() - j;
        let b_mag = v[start..end].to_vec();
        let b = BigInt::new(b_mag, 1);

        let result = a + b;

        unsafe {
            std::ptr::copy(result.mag.as_ptr().add(1), u.as_mut_ptr().add(start), n);
        }
    }
}

impl DivAssign for BigInt {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}

impl Div<&BigInt> for &BigInt {
    type Output = BigInt;

    fn div(self, rhs: &BigInt) -> Self::Output {
        self.clone() / rhs.clone()
    }
}

impl DivAssign<&BigInt> for BigInt {
    fn div_assign(&mut self, rhs: &BigInt) {
        *self = self.clone() / rhs.clone();
    }
}

impl Rem for BigInt {
    type Output = BigInt;

    fn rem(self, v: Self) -> Self::Output {
        let u_len = self.mag.len();
        let v_len = v.mag.len();
        if self.signum == 0 {
            return ZERO;
        } else if v.signum == 0 {
            panic!("Mod by zero.");
        } else if u_len == 1 && v_len == 1 {
            let u = self.mag[0] as i64 * self.signum as i64;
            let v = v.mag[0] as i64 * v.signum as i64;
            let r = u % v;
            if r == 0 { return ZERO; }
            let signum = if r < 0 { -1 } else { 1 };
            return BigInt::new(vec![r.abs() as u32], signum);
        } else if v_len == 1 {
            return BigInt::mod_one_word(self, v);
        } else if self < v {
            return self;
        } else {
            return BigInt::mod_knuth(self, v);
        }
    }
}

impl RemAssign for BigInt {
    fn rem_assign(&mut self, rhs: Self) {
        *self = self.clone() % rhs;
    }
}

// 实现求余
impl BigInt {
    fn mod_knuth(mut u: BigInt, mut v: BigInt) -> BigInt {
        let u_len = u.mag.len(); // m + n
        let v_len = v.mag.len(); // n
        let n = v_len;
        let m = u_len - v_len;

        // D1 normalize
        // Vn-1 * d >= b / 2
        // here b = 2^32
        // and  Vn-1 is u32
        // bits of Vn-1: [0..xxxx]
        // shift Vn-1 to [xxx0..0]
        // so Vn-1 >= [01000..0](2^31)
        let shift_len = v.mag[0].leading_zeros();
        if shift_len == 0 {
            u.mag.push(0);
        } else {
            u = u << shift_len;
            v = v << shift_len;
        }
        let m = if u.mag.len() <= n + m {
            m - 1
        } else {
            m
        };
        // D2
        // let j = m
        let b: u64  = 1 << 32;
        let u_len   = u.mag.len();
        for j in (0..=m).rev() {
            let sum = ((u.mag[u_len - 1 - n - j]   as u64) << 32) +
                            u.mag[u_len - n - j] as u64;
            // D3 calculate qhat
            let mut qhat: u64 = sum / (v.mag[0] as u64);
            let mut rhat: u64    = sum % (v.mag[0] as u64);
            // check
            while (rhat >> 32_u64) == 0 {
                if qhat != b {
                    let product = qhat * v.mag[1] as u64;
                    let sum = (rhat << 32_u64) + (u.mag[u_len + 1 - j - n] as u64);
                    if product > sum {
                        qhat -= 1;
                        rhat += v.mag[0] as u64;
                    } else {
                        break;
                    }
                } else {
                    qhat -= 1;
                    rhat += v.mag[0] as u64;
                }
            }
            let result = BigInt::mul_sub(&mut u.mag, &v.mag, qhat, n, j);
            if result < 0 {
                BigInt::add_back(&mut u.mag, &v.mag, n, j);
            }
        }
        BigInt::divide_one_word(u, BigInt::from(1_u32 << shift_len))
    }
    fn mod_one_word(mut num_u: BigInt, num_v: BigInt) -> BigInt {
        let u = &mut num_u.mag;
        let v = num_v.mag[0];

        let mut r = 0;
        let n = u.len();
        let mut w = new_zero_vec_with_cap!(n);

        for j in 0..n {
            let sum: u64 = (r << 32) + u[j] as u64;
            w[j] = (sum / (v as u64)) as u32;
            r = sum % (v as u64);
        }

        let signum = num_u.signum * num_v.signum;
        // check
        if r == 0 {
            return ZERO;
        } else {
            return BigInt::new(vec![r as u32], signum);
        }
    }
}

impl Rem<&BigInt> for &BigInt {
    type Output = BigInt;

    fn rem(self, rhs: &BigInt) -> Self::Output {
        self.clone() % rhs.clone()
    }
}

impl RemAssign<&BigInt> for BigInt {
    fn rem_assign(&mut self, rhs: &BigInt) {
        *self = self.clone() % rhs.clone();
    }
}

#[test]
fn test_from() {
    let num: i8 = 12;
    let big: BigInt = num.into();
    assert_eq!(big.mag[0] as i8, num * (big.signum as i8));

    let num: i16 = -100;
    let big: BigInt = num.into();
    assert_eq!(big.mag[0] as i16, num * (big.signum as i16));

    let num: i32 = 100;
    let big: BigInt = num.into();
    assert_eq!(big.mag[0] as i32, num * (big.signum as i32));

    let num: isize = -10000;
    let big: BigInt = num.into();
    assert_eq!(big.mag[0] as isize, num * (big.signum as isize));

    let num: i64 = -113132;
    let big: BigInt = num.into();
    assert_eq!(big.mag[0] as i64, num * (big.signum as i64));
}

#[test]
fn test_square() {
    // test square to len
    let three: BigInt = "3".into();
    assert_eq!(BigInt::from("9"), three.square());

    // test square_karatsuba
    let a1: BigInt = concat!(
        "19419855859737227032312055680651075480951739552367379972199925016867623661988",
        "89990696602287822477109056244119751485828053901775315360842915317053088973664",
        "00496619142218693948014537641790214007434607050498425882666156551281363510551",
        "11099518879143485548014640015583670639060917474447573958129564134328565451643",
        "68202413247373020246149526055842924756979535372710091822317187409687953220877",
        "01654988392543045341566871992242871769575404195527194369021160117319175829268",
        "04913684816055789157773349015897950557404165767898522572194900900562949054443",
        "63987477269054976500305750718540809303727028907777678293228091075833895367701",
        "99902578317870722364103253543749464393683466960206787641777673976220741106335",
        "37434817979366540532537695846069064538750075192677478668972402329824359901707",
        "63411539288338611798441205335955633573162839286510650396082861144710651642799",
        "48874072924254598020743480848878113888005642718749354518404986316558314872761",
        "4126591979881590052491002961475166519803"
    ).into();
    let real_a1_s: BigInt = concat!(
        "3771308016129703132872604173496236830174514076412540555223729096856875125677",
        "5750283416008642269789430086716381824264503900231300426593835969568743798302",
        "5526076917051233279339534751836085151342288574101724008074689600522133942192",
        "0923526468395717774696991075726622911991700415257783608251849620468353195352",
        "1188917997165703336536200114053763856875384699131864737313252680444840462252",
        "1536140375804053585260865941999153247548341185014534048373369617126714862480",
        "4192778070314041918110474714034081448541039646284093130162197352112423624453",
        "7251919257461014044136460911055967416341565120346660605685002452157726936906",
        "6993327283060999886725973302825811215195562500937917489663439561693528677526",
        "9484376709690633991045763833972079163827372920490052557671193766857777165755",
        "3693553744991822807990163072652813916800068385701138822396243022125952602104",
        "2078892805052767449330551265380793760691467738146861543737314378352335063518",
        "4914392081285586003884299991564006551930613595037764365377109230819488196311",
        "8722466476693151249440579716888574200624999560550849749880932616429908080942",
        "4888071270322512212753303452901883578760298024622902423262106422594151216580",
        "3980058916519521445894689396841163961064635773502056349956941444726845938760",
        "3197072124059673627102974186294215472728971061566110215329570266442920160939",
        "3786919542152316347785297843376940533149683206000990133761289824134988133403",
        "2151033765902574352685648512685022461873360043985963683407024102444263804490",
        "1999290111885525653033427086778966526167581397350578843681885992810403205947",
        "6752910780799580481763882121171195320572159945622740537314174125753679175288",
        "4884191061561602830809630383607901174528833542179649622384342387643851298429",
        "3340492917601706269229017810084377736148017862503019540479449478391820163131",
        "3558310564412415854452632812068878880176323955710774537330716856315759331947",
        "3317990924714636570792833361871098618337897324246863611061415195984926254955",
        "457120459494907694791158809",
    ).into();
    assert_eq!(real_a1_s, a1.square());

    // test square_toom_cook
    let c: BigInt = concat!(
        "10000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000"
    ).into();
    let e = c.clone() * c;
    let f = e.clone() * e;
    let real_f = concat!(
          "100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    ).into();
    assert_eq!(f, real_f);
}

#[test]
fn test_mul_by_int() {
    let a: BigInt = "10000000000000000".into();
    let b: BigInt = "3001".into();
    let result: BigInt = "30010000000000000000".into();
    assert_eq!(result, a.clone() * b.clone());
    assert_eq!(result, b * a);
}

#[test]
fn test_mul_to_len() {
    let a: BigInt = "10000000000000000".into();
    let b: BigInt = "30000000000000000".into();
    let result: BigInt = "300000000000000000000000000000000".into();
    assert_eq!(a * b, result);
}

#[test]
fn test_exact_divide_by_3() {
    let three: BigInt = "30000000000".into();
    let six: BigInt = "6".into();

    let result: BigInt = "10000000000".into();
    assert_eq!(result, three.exact_divide_by_3());
    assert_eq!(BigInt::from("2"), six.exact_divide_by_3());
}

#[test]
fn test_mul_karatsuba() {
    let a: BigInt = concat!(
        "1869453311573993456634306574393586908889230632398541952520898688168215923527",
        "7176197057068536250224262194068484309584118170511105189909575048793518893185",
        "7507727867495819246726159865769898262968667885355361791649192109358497086474",
        "7936932811373813774623125931906054980527575734607403872136608118256271348216",
        "6459826159031091038206102815727850329522533220259255633048039962519253047861",
        "7109155125846374166827926723777680219270510919194419535495591881004557835319",
        "5838135928895817041355184601039902939511193792968645840244341777472198081073",
        "6112415233959017344713438246965687099426741123625732029040774680291924647520",
        "0575843767951441831069571403178743390652879235414717708920676593287757457693",
        "6756848365339986906294790149667894755709309460295961128666353544554689947661",
        "0261858658621780074515448135017949427753259673674704352229778982732188869872",
        "9379135395499170327475724139509815885344218793528455279111575061198807382349",
        "9482352103252575426411932947891590878159765921340072",
    ).into();
    let b: BigInt = concat!(
        "1941985585973722703231205568065107548095173955236737997219992501686762366198",
        "8899906966022878224771090562441197514858280539017753153608429153170530889736",
        "6400496619142218693948014537641790214007434607050498425882666156551281363510",
        "5511109951887914348554801464001558367063906091747444757395812956413432856545",
        "1643682024132473730202461495260558429247569795353727100918223171874096879532",
        "2087701654988392543045341566871992242871769575404195527194369021160117319175",
        "8292680491368481605578915777334901589795055740416576789852257219490090056294",
        "9054443639874772690549765003057507185408093037270289077776782932280910758338",
        "9536770199902578317870722364103253543749464393683466960206787641777673976220",
        "7411063353743481797936654053253769584606906453875007519267747866897240232982",
        "4359901707634115392883386117984412053359556335731628392865106503960828611447",
        "1065164279948874072924254598020743480848878113888005642718749354518404986316",
        "5583148727614126591979881590052491002961475166519803",
    ).into();
    let result: BigInt = concat!(
        "3630451384727538085778504091915007826221192500788661241639309939327030557375",
        "5857499795452701020038058819844595919564402528410489156696859586497551044808",
        "9126293159001200884918879249281854422357193065160524969395783660261577660364",
        "1563857284576197856958710694751532590370737105723826324622518751130622578830",
        "7375163311414397966156794621731758382721070048584686666252216391003400602882",
        "4871659734613447348885753388170631216867217964110073602116749905278122844351",
        "7714515539662641364964498545951630896489933783706267084871530197279688940058",
        "1441261826623480183972455055316120085207822920650276846871823190600984361493",
        "4490292315494112877964653926739427892146972888369199652059072993191343592255",
        "8146029402732460885948737877628050570264032281397067775791020367842573077209",
        "5932737952436627910448963092923834330939939363705208407706034364683976762081",
        "3389362987418277477537608896286838338187495262852622885927152580293109160707",
        "3307132913992584224272170489734478146555962592400042245813429443825344956254",
        "5116587793273269480150842847677345656659136177728536527426618892538602074169",
        "2366185636420256365572002274531759381979508056574533481259020811851306033819",
        "2606040864540642196034689406464534660757858232146791098598133705368678969453",
        "7401066063004558982897051623607368721643611233894230332261230117361701457952",
        "6323750278980648200361979289810156156916579200502776573921765550349714434972",
        "6187569849709155710906187554118789437181709500629410362772422238756041128913",
        "6738292384748741416978674795131436668446615022763994321750709472435556073504",
        "2302594074130122589074206700925732064306605716071325057063073049368068436250",
        "5895053881024574764551795648675382733689642579382932268035301831750858483207",
        "2236881414980595224991331112440086612952839191754781367076429450970496413917",
        "2539785868552012593529150035317324456416534002806840140998850361657977019718",
        "3815546311605097764413944376726762720223549048679261530395160146483881383066",
        "712889013813473862285445816"
    ).into();
    assert_eq!(a * b, result);
}

#[test]
fn test_mul_toom_cook3() {
    let a: BigInt = concat!(
        "3000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    ).into();
    let b: BigInt = concat!(
        "100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    ).into();
    let result: BigInt = concat!(
        "300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    ).into();
    assert_eq!(a * b, result);
}

#[test]
fn test_div() {
    // test divide
    let a = BigInt::from("120");
    let b = BigInt::from("13");
    let c = BigInt::from("9");
    assert_eq!(a / b, c);
    // test divide one word
    let a = BigInt::from("10000000000000000000000000000000000");
    let b = BigInt::from("1000");
    let c = BigInt::from("10000000000000000000000000000000");
    assert_eq!(a / b, c);
    // test divide Knuth
    let a = BigInt::from("124871287894782164876238905710532895792830741278950327951074309571023759712087492109591287094780219747214567876543245678976547897654367543567654678987654321456789087654325678908765432567890876543245678908765432567890876543876543245678907654356789");
    let b = BigInt::from("5678987654678976543587654678976546789087657876545678976543256789765432456789234567890854376");
    let c = BigInt::from("21988300642263136800048566126805476040703295625345756336585704044222781621158596876349726562910906651562104831721609088222205401883168960593370061500432215");
    assert_eq!(a / b, c);
}

#[test]
fn test_mod() {
    let a: BigInt = "12".into();
    let b: BigInt = "8".into();
    let r: BigInt = "4".into();
    assert_eq!(a % b, r);

    let a: BigInt = "10000000000000000".into();
    let b: BigInt = "10".into();
    let r = ZERO;
    assert_eq!(a % b, r);

    let a: BigInt = "23456789873625348759607098765432345678909876325346546543456453573434839063464369876543245".into();
    let b: BigInt = "526738495607659438721653478560954837265378495607".into();
    let r: BigInt = "393707270751296419349581795408095683999332705291".into();
    assert_eq!(a % b, r);
}

#[test]
fn test_to_u64() {
    let a: BigInt = "123456789110".into();
    let b: u64 = 123456789110;
    assert_eq!(a.to_u64(), b);
}

#[test]
fn test_to_string() {
    let a: BigInt = "12345678909876523784950683472613487560983287654321".into();
    let b = a.small_to_string(10);
    let c = "12345678909876523784950683472613487560983287654321".to_string();
    assert_eq!(b, c, "b = {}", b);

    let a = BigInt::from_str_radix("fafcfbffaffaffbffbffbffbffbff", 16);
    let b = a.small_to_string(16);
    let c = "fafcfbffaffaffbffbffbffbffbff".to_string();
    assert_eq!(b, c, "b = {}", b);
    
    let a: BigInt = "12345678909876543213456789098574635425364758697096854736526458798765444375328977748874784874784874874587458742745748748745878389234734803142908342342983429834298324983429834219834983421983241983429834983429834219803249842198342983249328409832043892403429832948034289324890432890480943980342198493898342198429830423942834289342890".into();
    println!("{}", a.to_string());
}
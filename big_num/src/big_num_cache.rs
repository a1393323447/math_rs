use lazy_static::*;

use crate::BigInt;
use crate::big_int::ZERO;
use crate::big_num_constants::*;

lazy_static! {
    pub static ref POS_CACHE: [BigInt; MAX_CONSTANT + 1] = [
        unsafe { BigInt::from_raw(vec![ ] , 0) },
        unsafe { BigInt::from_raw(vec![1] , 1) },
        unsafe { BigInt::from_raw(vec![2] , 1) },
        unsafe { BigInt::from_raw(vec![3] , 1) },
        unsafe { BigInt::from_raw(vec![4] , 1) },
        unsafe { BigInt::from_raw(vec![5] , 1) },
        unsafe { BigInt::from_raw(vec![6] , 1) },
        unsafe { BigInt::from_raw(vec![7] , 1) },
        unsafe { BigInt::from_raw(vec![8] , 1) },
        unsafe { BigInt::from_raw(vec![9] , 1) },
        unsafe { BigInt::from_raw(vec![10], 1) },
        unsafe { BigInt::from_raw(vec![11], 1) },
        unsafe { BigInt::from_raw(vec![12], 1) },
        unsafe { BigInt::from_raw(vec![13], 1) },
        unsafe { BigInt::from_raw(vec![14], 1) },
        unsafe { BigInt::from_raw(vec![15], 1) },
        unsafe { BigInt::from_raw(vec![16], 1) },
    ];
    pub static ref NEG_CACHE: [BigInt; MAX_CONSTANT + 1] = [
        unsafe { BigInt::from_raw(vec![ ] ,  0) },
        unsafe { BigInt::from_raw(vec![1] , -1) },
        unsafe { BigInt::from_raw(vec![2] , -1) },
        unsafe { BigInt::from_raw(vec![3] , -1) },
        unsafe { BigInt::from_raw(vec![4] , -1) },
        unsafe { BigInt::from_raw(vec![5] , -1) },
        unsafe { BigInt::from_raw(vec![6] , -1) },
        unsafe { BigInt::from_raw(vec![7] , -1) },
        unsafe { BigInt::from_raw(vec![8] , -1) },
        unsafe { BigInt::from_raw(vec![9] , -1) },
        unsafe { BigInt::from_raw(vec![10], -1) },
        unsafe { BigInt::from_raw(vec![11], -1) },
        unsafe { BigInt::from_raw(vec![12], -1) },
        unsafe { BigInt::from_raw(vec![13], -1) },
        unsafe { BigInt::from_raw(vec![14], -1) },
        unsafe { BigInt::from_raw(vec![15], -1) },
        unsafe { BigInt::from_raw(vec![16], -1) },
    ];
    pub static ref LONG_RADIX: [BigInt;37] = [ ZERO, ZERO,
        BigInt::from(0x4000000000000000_u64), BigInt::from(0x383d9170b85ff80b_u64),
        BigInt::from(0x4000000000000000_u64), BigInt::from(0x6765c793fa10079d_u64),
        BigInt::from(0x41c21cb8e1000000_u64), BigInt::from(0x3642798750226111_u64),
        BigInt::from(0x1000000000000000_u64), BigInt::from(0x12bf307ae81ffd59_u64),
        BigInt::from( 0xde0b6b3a7640000_u64), BigInt::from(0x4d28cb56c33fa539_u64),
        BigInt::from(0x1eca170c00000000_u64), BigInt::from(0x780c7372621bd74d_u64),
        BigInt::from(0x1e39a5057d810000_u64), BigInt::from(0x5b27ac993df97701_u64),
        BigInt::from(0x1000000000000000_u64), BigInt::from(0x27b95e997e21d9f1_u64),
        BigInt::from(0x5da0e1e53c5c8000_u64), BigInt::from( 0xb16a458ef403f19_u64),
        BigInt::from(0x16bcc41e90000000_u64), BigInt::from(0x2d04b7fdd9c0ef49_u64),
        BigInt::from(0x5658597bcaa24000_u64), BigInt::from( 0x6feb266931a75b7_u64),
        BigInt::from( 0xc29e98000000000_u64), BigInt::from(0x14adf4b7320334b9_u64),
        BigInt::from(0x226ed36478bfa000_u64), BigInt::from(0x383d9170b85ff80b_u64),
        BigInt::from(0x5a3c23e39c000000_u64), BigInt::from( 0x4e900abb53e6b71_u64),
        BigInt::from( 0x7600ec618141000_u64), BigInt::from( 0xaee5720ee830681_u64),
        BigInt::from(0x1000000000000000_u64), BigInt::from(0x172588ad4f5f0981_u64),
        BigInt::from(0x211e44f7d02c1000_u64), BigInt::from(0x2ee56725f06e5c71_u64),
        BigInt::from(0x41c21cb8e1000000_u64)
    ];
    pub static ref LOG_CACHE: [f64; 37] = {
        let mut log_cache = [0_f64; 37];
        for i in 2..=36 {
            log_cache[i] = (i as f64).ln();
        }
        log_cache
    };
}
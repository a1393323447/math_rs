pub const DIGITS: [char; 36] = [
    '0' , '1' , '2' , '3' , '4' , '5' ,
    '6' , '7' , '8' , '9' , 'a' , 'b' ,
    'c' , 'd' , 'e' , 'f' , 'g' , 'h' ,
    'i' , 'j' , 'k' , 'l' , 'm' , 'n' ,
    'o' , 'p' , 'q' , 'r' , 's' , 't' ,
    'u' , 'v' , 'w' , 'x' , 'y' , 'z'
];

pub const DIGITS_PER_LONG: [usize; 37] = [0, 0,
    62, 39, 31, 27, 24, 22, 20, 19, 18, 18, 17, 17, 16, 16, 15, 15, 15, 14,
    14, 14, 14, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12
];

pub const BITS_PER_DIGIT: [usize; 37] =  [
    0, 0,
    1024, 1624, 2048, 2378, 2648, 2875, 3072, 3247, 3402, 3543, 3672,
    3790, 3899, 4001, 4096, 4186, 4271, 4350, 4426, 4498, 4567, 4633,
    4696, 4756, 4814, 4870, 4923, 4975, 5025, 5074, 5120, 5166, 5210,
    5253, 5295
];

pub const DIGITS_PER_INT: [usize; 37] = [
    0, 0, 
    30, 19, 15, 13, 11, 11, 10, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5
];

pub const INT_RADIX: [u32; 37] = [
    0, 0,
    0x40000000, 0x4546b3db, 0x40000000, 0x48c27395, 0x159fd800,
    0x75db9c97, 0x40000000, 0x17179149, 0x3b9aca00, 0xcc6db61,
    0x19a10000, 0x309f1021, 0x57f6c100, 0xa2f1b6f,  0x10000000,
    0x18754571, 0x247dbc80, 0x3547667b, 0x4c4b4000, 0x6b5a6e1d,
    0x6c20a40,  0x8d2d931,  0xb640000,  0xe8d4a51,  0x1269ae40,
    0x17179149, 0x1cb91000, 0x23744899, 0x2b73a840, 0x34e63b41,
    0x40000000, 0x4cfa3cc1, 0x5c13d840, 0x6d91b519, 0x39aa400
];

pub const MAX_CONSTANT: usize = 16;

pub const MAX_MAG_LENGTH: usize = isize::MAX as usize / isize::BITS as usize + 1;

pub const KARATSUBA_THRESHOLD: usize = 80;

pub const TOOM_COOK_THRESHOLD: usize = 240;

pub const KARATSUBA_SQUARE_THRESHOLD: usize = 128;

pub const TOOM_COOK_SQUARE_THRESHOLD: usize = 216;

pub const MULTIPLY_SQUARE_THRESHOLD: usize = 20;

pub const SCHOENHAGE_BASE_CONVERSION_THRESHOLD: usize = 20;
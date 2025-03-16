#![feature(stdarch_x86_avx512)]
#![feature(avx512_target_feature)]

use std::arch::x86_64::*;
use std::fs::{self, File};
use std::io::{BufReader, Read};

#[inline(always)]
fn hash_linear(str: &str) -> u32 {
    let mut h = 0u32;
    for c in str.bytes() {
        h = 37u32.wrapping_mul(h).wrapping_add(c as u32);
    }
    h
}

#[inline(always)]
fn hash_crc32_linear(str: &str) -> u32 {
    let mut string_bytes = [0u8; 64];

    for (i, byte) in str.bytes().enumerate() {
        string_bytes[i] = byte;
    }

    let mut crc = 0;
    for i in (0..64).step_by(8) {
        unsafe {
            crc = _mm_crc32_u64(
                crc,
                u64::from_ne_bytes(string_bytes[i..i + 8].try_into().unwrap_unchecked()),
            );
        }
    }

    crc as u32
}

macro_rules! hash_512 {
    ($s:ident) => {{
        let set0 = _mm512_castsi512_si256(_mm512_maskz_compress_epi64(0x0F, $s));
        let set1 = _mm512_castsi512_si256(_mm512_maskz_compress_epi64(0xF0, $s));

        let mut crc = 0;
        crc = _mm_crc32_u64(crc, _mm256_extract_epi64(set0, 0) as u64);
        crc = _mm_crc32_u64(crc, _mm256_extract_epi64(set0, 1) as u64);
        crc = _mm_crc32_u64(crc, _mm256_extract_epi64(set0, 2) as u64);
        crc = _mm_crc32_u64(crc, _mm256_extract_epi64(set0, 3) as u64);

        crc = _mm_crc32_u64(crc, _mm256_extract_epi64(set1, 0) as u64);
        crc = _mm_crc32_u64(crc, _mm256_extract_epi64(set1, 1) as u64);
        crc = _mm_crc32_u64(crc, _mm256_extract_epi64(set1, 2) as u64);
        crc = _mm_crc32_u64(crc, _mm256_extract_epi64(set1, 3) as u64);

        crc as u32
    }};
}

fn print_m256(name: &str, m: __m256i) {
    let mut arr = [0u8; 32];
    unsafe {
        _mm256_storeu_si256(arr.as_mut_ptr() as *mut _, m);
    }
    println!("{} {:x?}", name, arr);
}

fn print_m512(name: &str, m: __m512i) {
    let mut arr = [0u8; 64];
    unsafe {
        _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, m);
    }
    println!("{} {:?}", name, arr);
}

#[derive(Debug, Clone, Copy)]
struct CityTemp {
    // Todo: store city name
    // Todo: switch to smaller types, 999,-999 will fit in i16
    min: i32,
    max: i32,

    count: u32,
    sum: i64,
}

impl CityTemp {
    fn new() -> Self {
        Self {
            min: i32::MAX,
            max: i32::MIN,
            count: 0,
            sum: 0,
        }
    }

    #[inline(always)]
    fn add(&mut self, temp: i32) {
        self.min = self.min.min(temp);
        self.max = self.max.max(temp);
        self.count += 1;
        self.sum += temp as i64;
    }

    #[inline(always)]
    fn get_avg(&self) -> f64 {
        self.sum as f64 / self.count as f64 / 10.0
    }

    #[inline(always)]
    fn get_min(&self) -> f64 {
        self.min as f64 / 10.0
    }

    #[inline(always)]
    fn get_max(&self) -> f64 {
        self.max as f64 / 10.0
    }
}

// Todo: proper LUT size and optimise
const LUT_SIZE: usize = 0x406FF5F;
const CHUNK_SIZE: usize = 64; // 4 * 1024;
const CHUNK_ALIGN: usize = 64;

struct Worker {
    pub chunk: [u8; CHUNK_SIZE + CHUNK_ALIGN],
    pub lut: Box<[CityTemp; LUT_SIZE]>,
}

impl Worker {
    fn new() -> Self {
        let lut = vec![CityTemp::new(); LUT_SIZE]
            .into_boxed_slice()
            .try_into()
            .unwrap();

        Self {
            chunk: [0u8; CHUNK_SIZE + CHUNK_ALIGN],
            lut,
        }
    }
    // #[target_feature(
    //     enable = "avx,avx2,sse2,sse3,avx512f,avx512bw,avx512vl,avx512cd,avx512vbmi,avx512vbmi2,lzcnt"
    // )]
    #[inline(never)]
    unsafe fn process_chunk(&mut self, nread: usize) {
        unsafe {
            let semi_vec = _mm512_set1_epi8(';' as i8);
            let nl_vec = _mm512_set1_epi8('\n' as i8);
            let char0_vec = _mm512_set1_epi8('0' as i8);
            let char9_vec = _mm512_set1_epi8('9' as i8);
            let neg_vec = _mm512_set1_epi8('-' as i8);
            let mul_vec = _mm512_set1_epi32(0x00010A64);
            let ff_vec = _mm512_set1_epi32(-1);
            let cnst32_vec = _mm512_set_epi32(
                512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160, 128, 96, 64, 32,
            );

            // Read constraints - nread is CHUNK_SIZE or less. self.chunk is CHUNK_SIZE + CHUNK_ALIGN,
            // meaning that we can read up to 64 bytes into a mm512 past the end of the chunk. These reads
            // will fill the remaining of the register with 0s
            debug_assert!(nread <= CHUNK_SIZE);
            debug_assert!(self.chunk.len() <= CHUNK_SIZE + CHUNK_ALIGN);

            let mut chunk = &self.chunk[0..nread];

            // Read constraints - the chunk should end with a newline. Any extras should be
            // stored separately and prepended to the next chunk. Finally, the file ends in a new line.
            debug_assert!(chunk[chunk.len() - 1] == '\n' as u8);

            while chunk.len() > 0 {
                // Idea: instead of loading 64 bytes again, try processing 2 lines at a time
                // using shifts and avoiding branches. Technically we can try to do the same thing
                // again after shifting by line_len + 1 bytes, but need to make sure that we don't
                // submit the work if the next line is not complete.
                let in_vec = _mm512_loadu_si512(chunk.as_ptr() as *const _);

                // Todo: check the cost of trailing_zeros
                let line_len = _mm512_cmpeq_epi8_mask(in_vec, nl_vec).trailing_zeros();
                let name_len = _mm512_cmpeq_epi8_mask(in_vec, semi_vec).trailing_zeros();

                // Todo: Check if line_mask can be removed
                let line_mask = u64::MAX >> (64 - line_len);

                // Hashing
                let name_len_bits = name_len as i32 * 8;
                let shift = _mm512_subs_epu16(cnst32_vec, _mm512_set1_epi32(name_len_bits));
                let name_mask = _mm512_srlv_epi32(ff_vec, shift);

                let name_vec = _mm512_and_si512(in_vec, name_mask);

                let name_hash = hash_512!(name_vec);

                // Parse temperature
                let lt_mask = _mm512_cmp_epu8_mask(in_vec, char9_vec, _MM_CMPINT_LE);
                let gt_mask = _mm512_cmp_epu8_mask(in_vec, char0_vec, _MM_CMPINT_NLT);
                let digit_mask = _kand_mask64(_kand_mask64(lt_mask, gt_mask), line_mask);

                let digit_vec = _mm512_subs_epi8(in_vec, char0_vec);
                let digit_vec = _mm512_maskz_compress_epi8(digit_mask, digit_vec);

                let neg_mask = _mm512_cmpeq_epi8_mask(in_vec, neg_vec);
                let is_neg = ((neg_mask >> (name_len + 1)) & 1) as u32;

                // Shift digit mask to the right so it points to the byte before the first digit.
                // This will be 0b01010 for temperatures with 2 digits and 0b10110 for temperatures with 3 digits.
                // The first bit is always 0 (it's either the - sign or ; character depending if there is a sign character).
                let digit_mask_local = (digit_mask >> (name_len + is_neg)) as u32 as i32;

                // Pick bit 3 and use it directly as the shift amount
                // (e.g shift 8 bits for 2 digit numbers or 0 bits for 3 digit numbers)
                let shift = digit_mask_local & 0b1000;

                let shifted = _mm512_sllv_epi32(digit_vec, _mm512_set1_epi32(shift));

                let sum = _mm512_maddubs_epi16(shifted, mul_vec);
                let sum = _mm512_castsi128_si512(_mm_hadd_epi16(
                    _mm512_castsi512_si128(sum),
                    _mm512_castsi512_si128(sum),
                ));

                let temperature = [0u8; 4];
                _mm_storeu_si32(temperature.as_ptr() as *mut _, _mm512_castsi512_si128(sum));

                let neg = ((is_neg as i32) << 31) >> 31;
                let temperature = (i32::from_ne_bytes(temperature) ^ neg) - neg;

                // println!(
                //     "name_len={}, name_hash={} temperature={}",
                //     name_len, name_hash, temperature
                // );

                self.lut[name_hash as usize % LUT_SIZE].add(temperature);

                chunk = &chunk[line_len as usize + 1..];
            }

            assert!(chunk.is_empty());
        }
    }
}

struct ChunkReader {
    excess_len: usize,
    excess: [u8; 64], // 64 max line length
}

impl ChunkReader {
    pub fn new() -> Self {
        Self {
            excess_len: 0,
            excess: [0u8; 64],
        }
    }
}
impl ChunkReader {
    #[inline(always)]
    pub fn read_chunk<const S: usize>(&mut self, reader: &mut impl Read, out: &mut [u8]) -> usize {
        // Copy excess from a previous read to the start of the buffer
        out[..self.excess_len].copy_from_slice(&self.excess[..self.excess_len]);

        // Read new data after excess
        // Todo: check if the read chunk size can be made exact (e.g not depending on excess_len)
        let nread = reader.read(&mut out[self.excess_len..]).unwrap();

        // prev_excess_len is the length of the excess currently sitting at the start of the buffer
        let prev_excess_len = self.excess_len;

        // Calculate new newline cutoff for the current chunk
        self.excess_len = out
            .iter()
            .skip(prev_excess_len)
            .take(nread)
            .rev()
            .position(|&c| c == b'\n')
            .unwrap_or(0);

        // Copy new excess to the excess buffer
        let excess_start = prev_excess_len + nread - self.excess_len;
        self.excess[..self.excess_len]
            .copy_from_slice(&out[excess_start..excess_start + self.excess_len]);

        // Zero out bytes from the end of previous excess + current data
        // to the end of the buffer. Zeroing is important for vectorised processing.
        // Todo: if we manage to bottleneck on reading, check if its possible to send the length
        // and do zeroing with vectors.
        for z in excess_start..S {
            out[z] = 0;
        }

        excess_start
    }
}

fn main() {
    let mut worker = Worker::new();

    let f = File::open("data/sample16kb.txt").unwrap();
    let mut file_reader = BufReader::new(f);

    let mut chunk_reader = ChunkReader::new();

    loop {
        let nread = chunk_reader
            .read_chunk::<CHUNK_SIZE>(&mut file_reader, &mut worker.chunk[..CHUNK_SIZE]);

        if nread == 0 {
            break;
        }

        unsafe {
            worker.process_chunk(nread);
        }
    }

    for (i, city) in worker.lut.iter().enumerate() {
        if city.count != 0 {
            continue;
        }
        println!(
            "City {} [{}] - min: {}, max: {}, avg: {}",
            i,
            city.count,
            city.min,
            city.max,
            city.get_avg()
        );
    }

    println!("Done");
}

#[cfg(test)]
mod tests {
    use std::hash::{DefaultHasher, Hash, Hasher};

    use super::*;

    fn get_hash(vec: Vec<u8>) -> u64 {
        let mut hash = DefaultHasher::new();
        vec.hash(&mut hash);
        hash.finish()
    }

    fn begin_sample() -> (ChunkReader, BufReader<File>, u64) {
        const SAMPLE_PATH: &str = "data/sample16kb.txt";

        let f = File::open(SAMPLE_PATH).unwrap();
        let file_reader = BufReader::new(f);

        let hash = get_hash(fs::read(SAMPLE_PATH).unwrap());

        let chunk_reader = ChunkReader::new();

        (chunk_reader, file_reader, hash)
    }

    fn reconstruct_test<const S: usize>() {
        let mut chunk = [0u8; S];

        let (mut chunk_reader, mut file_reader, sample_hash) = begin_sample();

        let mut reconstructed = Vec::new();
        loop {
            let read_size = chunk_reader.read_chunk(&mut file_reader, &mut chunk);

            reconstructed.extend_from_slice(&chunk[..read_size]);

            if read_size == 0 {
                break;
            }
        }

        assert_eq!(get_hash(reconstructed), sample_hash, "hash mismatch");
    }

    #[test]
    fn test_chunked_reading_10() {
        reconstruct_test::<10>();
    }

    #[test]
    fn test_chunked_reading_32() {
        reconstruct_test::<32>();
    }

    #[test]
    fn test_chunked_reading_64() {
        reconstruct_test::<64>();
    }

    #[test]
    fn test_chunked_reading_512() {
        reconstruct_test::<512>();
    }

    #[test]
    fn test_chunked_reading_1024() {
        reconstruct_test::<1024>();
    }

    #[test]
    fn test_chunked_reading_4096() {
        reconstruct_test::<4096>();
    }
}

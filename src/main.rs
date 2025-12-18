#![feature(generic_const_exprs)]

use std::arch::x86_64::*;
use std::borrow::Cow;
use std::fs::File;
use std::io::Write;

mod fastmap;
mod fastqueue;
mod mmap;

const NUM_WORKERS: usize = 32;

#[inline(always)]
fn name_hash(name: &[u8]) -> u32 {
    if name.len() > 64 {
        let mut crc = 0;
        let (chunks, remainder) = name.as_chunks::<8>();

        for part in chunks {
            unsafe {
                crc = _mm_crc32_u64(crc, u64::from_ne_bytes(*part));
            }
        }

        for &byte in remainder {
            unsafe {
                crc = _mm_crc32_u8(crc as u32, byte) as u64;
            }
        }

        return crc as u32;
    }

    unsafe {
        let load_mask = (1u64 << name.len()) - 1;
        let input = _mm512_maskz_loadu_epi8(load_mask, name.as_ptr() as *const _);
        name_hash_avx512(input)
    }
}

#[inline(always)]
fn name_hash_avx512(input: __m512i) -> u32 {
    unsafe {
        let mut data = [0u64; 8];
        _mm512_storeu_si512(data.as_mut_ptr() as *mut _, input);

        let mut crc = 0;

        crc = _mm_crc32_u64(crc, data[0]);
        crc = _mm_crc32_u64(crc, data[1]);
        crc = _mm_crc32_u64(crc, data[2]);
        crc = _mm_crc32_u64(crc, data[3]);

        crc = _mm_crc32_u64(crc, data[4]);
        crc = _mm_crc32_u64(crc, data[5]);
        crc = _mm_crc32_u64(crc, data[6]);
        crc = _mm_crc32_u64(crc, data[7]);

        crc as u32
    }
}

#[derive(Debug, Clone, Copy)]
struct StationAggregate<'a> {
    min: i16,
    max: i16,

    count: u32,
    sum: i32,

    name: &'a [u8],
    name_len: u8,
}

impl Default for StationAggregate<'_> {
    fn default() -> Self {
        Self {
            min: i16::MAX,
            max: i16::MIN,
            count: 0,
            sum: 0,
            name: &[0u8; 0],
            name_len: 0,
        }
    }
}

const S: usize = std::mem::size_of::<StationAggregate>();

impl StationAggregate<'_> {
    #[inline(always)]
    fn add(&mut self, temp: i16) {
        // min/max should generate into cmovge/cmovle
        self.min = self.min.min(temp);
        self.max = self.max.max(temp);
        self.count += 1;
        self.sum += temp as i32;
    }

    #[inline(always)]
    fn collapse(&mut self, &other: &Self) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.count += other.count;
        self.sum += other.sum;
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

    #[inline(always)]
    fn get_name(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(&self.name[..self.name_len as usize])
    }
}

struct StationMap<'a> {
    map: fastmap::FastMap<{ Self::BUCKET_BITS }, { Self::SLOT_BITS }, StationAggregate<'a>>,
}

impl<'a> StationMap<'a> {
    const BUCKET_BITS: usize = 13;
    const SLOT_BITS: usize = 4;

    #[inline(always)]
    pub fn get_mut(
        &mut self,
    ) -> &mut fastmap::FastMap<{ Self::BUCKET_BITS }, { Self::SLOT_BITS }, StationAggregate<'a>>
    {
        &mut self.map
    }

    #[inline(always)]
    pub fn get(
        &self,
    ) -> &fastmap::FastMap<{ Self::BUCKET_BITS }, { Self::SLOT_BITS }, StationAggregate<'a>> {
        &self.map
    }

    #[inline(always)]
    fn hash_u32(val: u32) -> u32 {
        // Black magic seed for optimal distribution
        val.wrapping_mul(0x5bc9d60a) >> (32 - Self::BUCKET_BITS)
    }
}

struct Worker<'a> {
    buf: &'a [u8],
    map: StationMap<'a>,
}

impl<'a> Worker<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            map: StationMap {
                map: fastmap::FastMap::new(),
            },
        }
    }

    #[inline(never)]
    fn process_line_slow(&mut self, chunk: &'a [u8]) -> usize {
        let mut name_len = 0;
        let mut line_len = 0;
        let mut temp = 0i16;

        for i in 0..chunk.len() {
            let ch = chunk[i];
            match ch {
                b'\n' => {
                    line_len = i;
                    let sign = chunk[name_len + 1] == b'-';

                    for j in chunk[name_len + 1..line_len].iter().enumerate() {
                        match j.1 {
                            b'0'..=b'9' => {
                                temp = temp * 10 + (j.1 - b'0') as i16;
                            }
                            _ => {}
                        }
                    }

                    if sign {
                        temp = -temp;
                    }

                    break;
                }
                b';' => {
                    name_len = i;
                }
                _ => {}
            }
        }

        let name_hash = name_hash(&chunk[..name_len]);
        let bucket = StationMap::hash_u32(name_hash);

        unsafe {
            let entry: &mut StationAggregate<'_> = self.map.get_mut().get_insert(name_hash, bucket);

            entry.name = &chunk[..name_len];
            entry.name_len = name_len as u8;
            entry.add(temp);
        }

        line_len
    }

    #[target_feature(
        enable = "avx,avx2,sse2,sse3,sse4.2,avx512f,avx512bw,avx512vl,avx512cd,avx512vbmi,avx512vbmi2,bmi1,popcnt"
    )]
    #[inline(never)]
    unsafe fn run(&mut self) {
        unsafe {
            // Prepare vectors that contain characters for parsing
            let semi_vec = _mm512_set1_epi8(';' as i8);
            let nl_vec = _mm512_set1_epi8('\n' as i8);
            let char0_vec = _mm512_set1_epi8('0' as i8);
            let char9_vec = _mm512_set1_epi8('9' as i8);
            let neg_vec = _mm512_set1_epi8('-' as i8);
            let zero_vec = _mm512_setzero_si512();

            // Temperatures are floating points with a single decimal point. Parsing
            // will take advantage of this by compressing the digits into 32-bits which
            // can be vertically multiplied by this vector to get an integer value for
            // the final temperature. Temperature integer can then be divided by 10 during
            // aggregation to get the final floating point value.
            let mul_vec = _mm512_set1_epi32(0x00010A64);

            let mut chunk = self.buf;

            // Main processing loop. The loop will read 512 bits from the input chunk and parse
            // the station name and temperature from it using AVX-512 instructions to process 64
            // bytes (a full row) at a time. In the ideal scenario the body of the loop gets compiled
            // to a branchess block of instructions with a single jump at the end to start on the next line
            while !chunk.is_empty() {
                // Read the next 512-bits from the input chunk into in_vec. The chunk buffer contains
                // a zero-filled 64 bytes padding at the end which may be read to the vector
                let in_vec = _mm512_loadu_si512(chunk.as_ptr() as *const _);

                // Test against special characters (; for name end and \n for line end) and
                // find the lengths of the corresponding components in the input by counting trailing ones
                let line_len = _mm512_cmpneq_epi8_mask(in_vec, nl_vec).trailing_ones();
                let name_len = _mm512_cmpneq_epi8_mask(in_vec, semi_vec).trailing_ones();

                if line_len == 64 {
                    let line_len = self.process_line_slow(chunk);

                    #[cfg(feature = "safety_checks")]
                    {
                        chunk = &chunk[line_len as usize + 1..];
                    }
                    #[cfg(not(feature = "safety_checks"))]
                    {
                        chunk = &chunk.get_unchecked(line_len as usize + 1..);
                    }

                    continue;
                }

                // Create a mask that is 1s for the station name portion of the line
                let name_line_mask = 0xFFFFFFFF_FFFFFFFFu64 >> (64 - name_len);

                // Create a mask that is 1s for the portion of the line that does not belong to the station
                // name, including ";-." characters (inside the quotes) and the temperature reading
                let rest_line_mask = ((1 << (line_len - name_len)) - 1) << name_len;

                // Generate a hash for the name. Bits that are not part of the station name are set to 0
                let name_vec = _mm512_mask_mov_epi8(zero_vec, name_line_mask, in_vec);
                let name_hash = name_hash_avx512(name_vec);

                // Parse temperature. A digit mask will be constructed that contains
                // 1 for the bytes in the input that are ascii digits 0 to 9
                let lt_mask = _mm512_cmp_epu8_mask(in_vec, char9_vec, _MM_CMPINT_LE);
                let gt_mask = _mm512_cmp_epu8_mask(in_vec, char0_vec, _MM_CMPINT_NLT);
                let digit_mask = _kand_mask64(_kand_mask64(lt_mask, gt_mask), rest_line_mask);

                // Extract digits from the input and compress them into a new vector. The low
                // 16 or 32 bits of the vector (depending if the input has 2 or 3 digits)
                // will contain the digits (in numeric, not ascii form) and the rest will be 0s
                let digit_vec = _mm512_subs_epi8(in_vec, char0_vec);
                let digit_vec = _mm512_maskz_compress_epi8(digit_mask, digit_vec);

                // Check if the line contains a "-" character and compress the mask into the bottom bit of is_neg
                let is_negative =
                    (_mm512_mask_cmpeq_epi8_mask(rest_line_mask, in_vec, neg_vec) != 0) as u32;

                // Shift digit mask to the right so it points to the byte before the first digit.
                // This will be 0b01010 for temperatures with 2 digits and 0b10110 for temperatures with 3 digits.
                // The first bit is always 0 (it's either the - sign or ; character depending if there is a sign character)
                let digit_mask_local = (digit_mask >> (name_len + is_negative)) as u32 as i32;

                // Pick bit 3 and use it directly as the shift amount
                // (e.g shift 8 bits for 2 digit numbers or 0 bits for 3 digit numbers)
                let shift = digit_mask_local & 0b1000;

                // Shift the digit vector right 8 bits if the temperature has 2 digits. This makes
                // sure that the 2 least significant digits are always in bits 8..23 of the vector
                let shifted = _mm512_sllv_epi32(digit_vec, _mm512_set1_epi32(shift));

                // Multiply 8-bit digits to the correct order of magnitude based on their bit position (producing 16-bit integers)
                // and add those 16-bit integers horizontally. E.g with input (order lsb..msb) [1, 2, 3, 0, ...]:
                // 1. Vertical multiply: [1, 2, 3, 0, ...] * [100, 10, 1, 0] -> [100, 20, 3, 0, ...]
                // 2. Horizont addition: [100, 20, 3, 0, ...] -> [120, 0, 3, 0, ...]
                let sum = _mm512_castsi512_si128(_mm512_maddubs_epi16(shifted, mul_vec));

                // Horizontally add bits 0..15 and 16..31 to create a single 16-bit temperature value:
                // [120, 0, 3, 0, ...] -> [123, 0, 0, 0, ...]
                let sum = _mm_hadd_epi16(sum, sum);

                // Load the 16-bit temperature value into a 32-bit integer for sign processing & aggregation
                // (it could be loaded to a 16-bit integer directly which seemed to give slightly worse code gen)
                let mut temperature = [0u8; 4];
                _mm_storeu_si32(temperature.as_mut_ptr() as *mut _, sum);

                // broadcast sign bit into a full integer and use xor to flip the bits
                // to make the temperature negative in 2's complement
                let neg = ((is_negative as i32) << 31) >> 31;
                let temperature = (i32::from_ne_bytes(temperature) ^ neg) - neg;

                let bucket = StationMap::hash_u32(name_hash);
                let entry = self.map.get_mut().get_insert(name_hash, bucket);

                // Copy the name vector to the entry. Most of the time it is
                // already there, but the store is cheap and avoids branching
                entry.name = chunk.as_ref();
                entry.name_len = name_len as u8;

                // According to spec temperature must be between -99.9 and 99.9
                debug_assert!(temperature >= -999 && temperature <= 999);
                entry.add(temperature as i16);

                #[cfg(feature = "safety_checks")]
                {
                    chunk = &chunk[line_len as usize + 1..];
                }
                #[cfg(not(feature = "safety_checks"))]
                {
                    chunk = &chunk.get_unchecked(line_len as usize + 1..);
                }
            }
        }
    }
}

#[target_feature(enable = "avx512f,sse4.2")]
unsafe fn process_stations<'a>(
    f: &'a File,
    station_map: &'a mut StationMap<'a>,
) -> Vec<(String, &'a StationAggregate<'a>)> {
    let buf = mmap::mmap_file(f).unwrap();

    let results = std::thread::scope(|s| {
        let chunk_size = buf.len() / NUM_WORKERS;

        let worker_threads =
            (0..NUM_WORKERS).fold((Vec::with_capacity(NUM_WORKERS), 0usize), |mut acc, _| {
                let mut split_at = (acc.1 + chunk_size).min(buf.len());

                for i in split_at..buf.len() {
                    if buf[i] == b'\n' {
                        split_at = i + 1;
                        break;
                    }
                }

                let mut worker = Worker::new(&buf[acc.1..split_at]);

                acc.0.push(s.spawn(move || {
                    unsafe {
                        worker.run();
                    }

                    return worker.map;
                }));
                acc.1 = split_at;

                return acc;
            });

        let results = worker_threads
            .0
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();

        results
    });

    for result in &results {
        for slot in result.get().backing.iter() {
            if slot.count == 0 {
                continue;
            }

            let entry = unsafe {
                let hash = name_hash(&slot.name[..slot.name_len as usize]);

                station_map
                    .get_mut()
                    .get_insert(hash, StationMap::hash_u32(hash))
            };

            if entry.count == 0 {
                *entry = *slot;
            } else {
                entry.collapse(&slot);
            }
        }
    }

    let mut stations = station_map
        .get()
        .backing
        .iter()
        .filter(|slot| slot.count > 0)
        .map(|slot| (slot.get_name().into_owned(), slot))
        .collect::<Vec<_>>();

    // Output must be alphabetically sorted
    stations.sort_by(|a, b| a.0.cmp(&b.0));

    stations
}

fn main() {
    let start_time = std::time::Instant::now();
    const FILE_PATH: &str = "data/measurements.txt";

    let mut station_map = StationMap {
        map: fastmap::FastMap::new(),
    };

    let f = File::open(FILE_PATH).unwrap();

    let stations = unsafe { process_stations(&f, &mut station_map) };

    let mut buf_writer = std::io::BufWriter::new(std::io::stdout().lock());

    buf_writer.write(b"{").unwrap();

    let num_stations = stations.len();
    for (ii, result) in stations.into_iter().enumerate() {
        let (name, station) = result;
        let name = name.trim_end_matches('\0');
        buf_writer
            .write_fmt(format_args!(
                "{name}={:.1}/{:.1}/{:.1}{}",
                station.get_min(),
                station.get_avg(),
                station.get_max(),
                if ii == num_stations - 1 { "" } else { ", " },
            ))
            .unwrap();
    }
    buf_writer.write(b"}\n").unwrap();

    buf_writer.flush().unwrap();

    println!("Took: {}ms", start_time.elapsed().as_millis());
    // println!("Done");
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        fs,
        io::{BufRead, BufReader},
    };

    use super::*;

    const TEST100B_PATH: &str = "data/test100b.txt";
    const SAMPLE_PATH: &str = "data/sample16kb.txt";
    const STATIONS_PATH: &str = "data/weather_stations.csv";

    fn naive<'a>(path: &str, mut name_buf: &'a mut [u8]) -> BTreeMap<String, StationAggregate<'a>> {
        let mut index: BTreeMap<String, StationAggregate<'a>> = BTreeMap::new();

        let f = File::open(path).unwrap();
        let mut file_reader = BufReader::new(f);

        let mut buf = String::new();
        loop {
            buf.clear();
            let nread = file_reader.read_line(&mut buf).unwrap();

            if nread == 0 {
                break;
            }

            'top: for (ii, c) in buf.bytes().enumerate() {
                match c {
                    b';' => {
                        let name_len = ii;

                        assert!(
                            name_len <= 100,
                            "station name \"{}\" ({} characters) too long",
                            &buf[..name_len],
                            name_len
                        );

                        let temp = buf[name_len + 1..].trim().parse::<f64>().unwrap();

                        let name = buf[..name_len].to_string();

                        let entry = index
                            .entry(name.clone())
                            .or_insert(StationAggregate::default());

                        entry.add((temp * 10.0) as i16);

                        if entry.name_len == 0 {
                            for (i, &b) in name.as_bytes().iter().enumerate() {
                                name_buf[i] = b;
                            }

                            let (n, rest) = name_buf.split_at_mut(name_len);

                            name_buf = rest;

                            entry.name = n;
                            entry.name_len = name_len as u8;
                        }

                        break 'top;
                    }
                    _ => {}
                }
            }
        }

        index
    }

    #[test]
    fn test_correctness() {
        let mut name_buf = vec![0u8; 1 * 1024 * 1024].into_boxed_slice();
        let naive_result = naive(SAMPLE_PATH, &mut name_buf);

        let mut station_map = StationMap {
            map: fastmap::FastMap::new(),
        };

        let f = File::open(SAMPLE_PATH).unwrap();

        let stations = unsafe { process_stations(&f, &mut station_map) };

        let mut result_count = 0;
        for (i, (station_name, station)) in stations.into_iter().enumerate() {
            result_count += 1;

            let naive_entry = naive_result.get(station_name.trim_end_matches('\0'));

            assert!(
                naive_entry.is_some(),
                "station {} not found in naive result (name={})",
                i,
                station_name,
            );
            let naive_station = naive_entry.unwrap();

            assert!(
                station.count == naive_station.count,
                "count mismatch ({}): optimised({}) != naive({})",
                naive_station.get_name(),
                station.count,
                naive_station.count
            );

            assert!(
                station.get_min() == naive_station.get_min(),
                "min mismatch ({}): optimised({}) != naive({})",
                naive_station.get_name(),
                station.get_min(),
                naive_station.get_min()
            );

            assert!(
                station.get_max() == naive_station.get_max(),
                "max mismatch ({}): optimised({}) != naive({})",
                naive_station.get_name(),
                station.get_max(),
                naive_station.get_max()
            );

            assert!(
                (station.get_avg() - naive_station.get_avg()).abs() < 0.0001,
                "avg mismatch ({}): optimised({}) != naive({})",
                naive_station.get_name(),
                station.get_avg(),
                naive_station.get_avg()
            );
        }

        // Check that naive result is sensible
        assert!(
            naive_result.len() == 983,
            "naive result count mismatch, got {}",
            naive_result.len()
        );

        assert!(
            result_count == naive_result.len(),
            "result count mismatch: {} != {}",
            result_count,
            naive_result.len()
        );
    }

    #[test]
    fn test_100b() {
        let mut name_buf = vec![0u8; 1 * 1024 * 1024].into_boxed_slice();
        let naive_result = naive(TEST100B_PATH, &mut name_buf);

        let mut station_map = StationMap {
            map: fastmap::FastMap::new(),
        };

        let f = File::open(TEST100B_PATH).unwrap();

        let stations = unsafe { process_stations(&f, &mut station_map) };

        let mut result_count = 0;
        for (i, (station_name, station)) in stations.into_iter().enumerate() {
            result_count += 1;

            let naive_entry = naive_result.get(station_name.trim_end_matches('\0'));

            assert!(
                naive_entry.is_some(),
                "station {} not found in naive result (name={})",
                i,
                station_name,
            );
            let naive_station = naive_entry.unwrap();

            assert!(
                station.count == naive_station.count,
                "count mismatch ({}): optimised({}) != naive({})",
                naive_station.get_name(),
                station.count,
                naive_station.count
            );

            assert!(
                station.get_min() == naive_station.get_min(),
                "min mismatch ({}): optimised({}) != naive({})",
                naive_station.get_name(),
                station.get_min(),
                naive_station.get_min()
            );

            assert!(
                station.get_max() == naive_station.get_max(),
                "max mismatch ({}): optimised({}) != naive({})",
                naive_station.get_name(),
                station.get_max(),
                naive_station.get_max()
            );

            assert!(
                (station.get_avg() - naive_station.get_avg()).abs() < 0.0001,
                "avg mismatch ({}): optimised({}) != naive({})",
                naive_station.get_name(),
                station.get_avg(),
                naive_station.get_avg()
            );
        }

        assert!(
            result_count == naive_result.len(),
            "result count mismatch: {} != {}",
            result_count,
            naive_result.len()
        );
    }

    #[test]
    fn test_hashing() {
        const WORKER_BUCKET_SLOTS: usize = 1 << StationMap::SLOT_BITS;

        let weather_stations = fs::read(STATIONS_PATH).unwrap();

        let mut weather_stations = weather_stations
            .split(|&c| c == b'\n')
            .map(|row| String::from_utf8(row.to_vec()).unwrap())
            .filter(|row| !row.starts_with("#"))
            .filter_map(|row| {
                let mut split = row.split(';');
                let name = split.next().unwrap();

                if name.is_empty() {
                    return None;
                }

                Some(name.to_string())
            })
            .collect::<Vec<_>>();

        weather_stations.sort();
        weather_stations.dedup();

        let mut index: BTreeMap<u32, usize> = BTreeMap::new();

        for station_name in &weather_stations {
            let h = StationMap::hash_u32(name_hash(station_name.as_bytes()));

            let entry = index.entry(h).or_insert(0);
            *entry += 1;

            assert!(
                *entry <= WORKER_BUCKET_SLOTS,
                "too many hash collisions for hash {} ({})",
                h,
                station_name
            );
        }
    }
}

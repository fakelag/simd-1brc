#![feature(stdarch_x86_avx512)]
#![feature(avx512_target_feature)]
#![feature(generic_const_exprs)]

use std::arch::x86_64::*;
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufReader, Read};

use workset::WorkSet;

mod fastmap;
mod workset;

const NUM_WORKERS: usize = 8;

macro_rules! hash_512 {
    ($s:ident) => {{
        let mut data = [0u64; 8];
        _mm512_storeu_si512(data.as_mut_ptr() as *mut _, $s);

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

fn print_m512_string(name: &str, m: __m512i) {
    let mut arr = [0u8; 64];
    unsafe {
        _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, m);
    }
    println!("{} {:?}", name, String::from_utf8_lossy(&arr));
}

#[derive(Debug, Clone, Copy)]
struct StationAggregate {
    min: i16,
    max: i16,

    // Todo: check the dataset to see if any of these types can be shrunk
    count: u32,
    sum: i64,

    name: [u8; 64],
}

impl Default for StationAggregate {
    fn default() -> Self {
        Self {
            min: i16::MAX,
            max: i16::MIN,
            count: 0,
            sum: 0,
            name: [0u8; 64],
        }
    }
}

impl StationAggregate {
    fn new() -> Self {
        Self {
            min: i16::MAX,
            max: i16::MIN,
            count: 0,
            sum: 0,
            name: [0u8; 64],
        }
    }

    #[inline(always)]
    fn add(&mut self, temp: i16) {
        self.min = self.min.min(temp);
        self.max = self.max.max(temp);
        self.count += 1;
        self.sum += temp as i64;
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
        // Todo: there is a faster from_utf8_unchecked that does not check for valid utf8
        String::from_utf8_lossy(&self.name)
    }
}

// Chunk align is the maximum input line length
const CHUNK_ALIGN: usize = 64;
const CHUNK_SIZE: usize = 4 * 1024 * 1024;

struct StationMap {
    map: fastmap::FastMap<{ Self::BUCKET_BITS }, { Self::SLOT_BITS }, StationAggregate>,
}

impl StationMap {
    const BUCKET_BITS: usize = 13;
    const SLOT_BITS: usize = 4;

    #[inline(always)]
    pub fn get_mut(
        &mut self,
    ) -> &mut fastmap::FastMap<{ Self::BUCKET_BITS }, { Self::SLOT_BITS }, StationAggregate> {
        &mut self.map
    }

    #[inline(always)]
    pub fn get(
        &self,
    ) -> &fastmap::FastMap<{ Self::BUCKET_BITS }, { Self::SLOT_BITS }, StationAggregate> {
        &self.map
    }

    #[inline(always)]
    fn hash_u32(val: u32) -> u32 {
        // Black magic seed for optimal distribution
        val.wrapping_mul(0x5bc9d60a) >> (32 - Self::BUCKET_BITS)
    }
}

struct Batch {
    pub chunk: Box<[u8; CHUNK_SIZE + CHUNK_ALIGN]>,
}

impl Default for Batch {
    fn default() -> Self {
        Self {
            chunk: vec![0u8; CHUNK_SIZE + CHUNK_ALIGN]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
        }
    }
}

struct Worker<'a> {
    pub set: &'a workset::WorkSet<NUM_WORKERS, Batch>,
    pub index: usize,
    pub map: StationMap,
}

impl<'a> Worker<'a> {
    pub fn new(index: usize, set: &'a workset::WorkSet<NUM_WORKERS, Batch>) -> Self {
        Self {
            set,
            index,
            map: StationMap {
                map: fastmap::FastMap::new(),
            },
        }
    }

    #[target_feature(
        enable = "avx,avx2,sse2,sse3,sse4.2,avx512f,avx512bw,avx512vl,avx512cd,avx512vbmi,avx512vbmi2,bmi1,popcnt"
    )]
    #[inline(never)] // Todo: remove
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

            loop {
                // Todo: set.start to return a guard with Drop impl that commits the set for safety
                let (nread, batch) = match self.set.start(self.index) {
                    Some(batch) => batch,
                    None => break,
                };
                let nread = nread as usize;

                // Read constraints - nread is CHUNK_SIZE or less. self.chunk is CHUNK_SIZE + CHUNK_ALIGN,
                // meaning that we can read up to 64 bytes into a mm512 past the end of the chunk. These reads
                // will fill the remaining of the register with 0s
                debug_assert!(nread <= CHUNK_SIZE);
                debug_assert!(batch.chunk.len() <= CHUNK_SIZE + CHUNK_ALIGN);

                let mut chunk = &batch.chunk[0..nread];

                // Read constraints - the chunk should end with a newline. Any extras should be
                // stored separately and prepended to the next chunk. Finally, the file ends in a new line
                debug_assert!(!chunk.is_empty());
                debug_assert!(chunk[chunk.len() - 1] == '\n' as u8);

                loop {
                    // Read the next 512-bits from the input chunk into in_vec. The chunk buffer contains
                    // a zero-filled 64 bytes padding at the end which may be read to the vector
                    let in_vec = _mm512_loadu_si512(chunk.as_ptr() as *const _);

                    // Test against special characters (; for name end and \n for line end) and
                    // find the lengths of the corresponding components in the input by counting trailing ones
                    let line_len = _mm512_cmpneq_epi8_mask(in_vec, nl_vec).trailing_ones();
                    let name_len = _mm512_cmpneq_epi8_mask(in_vec, semi_vec).trailing_ones();

                    // Create a mask that is 1s for the station name portion of the line
                    let name_line_mask = 0xFFFFFFFF_FFFFFFFFu64 >> (64 - name_len);

                    // Create a mask that is 1s for the portion of the line that does not belong to the station
                    // name, including ";-." characters (inside the quotes) and the temperature reading
                    let rest_line_mask = ((1 << (line_len - name_len)) - 1) << name_len;

                    // Generate a hash for the name. Bits that are not part of the station name are set to 0
                    let name_vec = _mm512_mask_mov_epi8(zero_vec, name_line_mask, in_vec);
                    let name_hash = hash_512!(name_vec);

                    // Parse temperature
                    let lt_mask = _mm512_cmp_epu8_mask(in_vec, char9_vec, _MM_CMPINT_LE);
                    let gt_mask = _mm512_cmp_epu8_mask(in_vec, char0_vec, _MM_CMPINT_NLT);
                    let digit_mask = _kand_mask64(_kand_mask64(lt_mask, gt_mask), rest_line_mask);

                    let digit_vec = _mm512_subs_epi8(in_vec, char0_vec);
                    let digit_vec = _mm512_maskz_compress_epi8(digit_mask, digit_vec);

                    let neg_mask = _mm512_cmpeq_epi8_mask(in_vec, neg_vec);
                    let is_neg = ((neg_mask >> (name_len + 1)) & 1) as u32;

                    // Shift digit mask to the right so it points to the byte before the first digit.
                    // This will be 0b01010 for temperatures with 2 digits and 0b10110 for temperatures with 3 digits.
                    // The first bit is always 0 (it's either the - sign or ; character depending if there is a sign character)
                    let digit_mask_local = (digit_mask >> (name_len + is_neg)) as u32 as i32;

                    // Pick bit 3 and use it directly as the shift amount
                    // (e.g shift 8 bits for 2 digit numbers or 0 bits for 3 digit numbers)
                    let shift = digit_mask_local & 0b1000;

                    let shifted = _mm512_sllv_epi32(digit_vec, _mm512_set1_epi32(shift));

                    let sum = _mm512_castsi512_si128(_mm512_maddubs_epi16(shifted, mul_vec));
                    let sum = _mm_hadd_epi16(sum, sum);

                    let mut temperature = [0u8; 4];
                    _mm_storeu_si32(temperature.as_mut_ptr() as *mut _, sum);

                    let neg = ((is_neg as i32) << 31) >> 31;
                    let temperature = (i32::from_ne_bytes(temperature) ^ neg) - neg;

                    let bucket = StationMap::hash_u32(name_hash);

                    let entry = self.map.get_mut().get_insert(name_hash, bucket);

                    _mm512_storeu_si512(entry.name.as_mut_ptr() as *mut _, name_vec);

                    // According to spec temperature must be between -99.9 and 99.9
                    debug_assert!(temperature >= -999 && temperature <= 999);
                    entry.add(temperature as i16);

                    chunk = &chunk[line_len as usize + 1..];

                    if chunk.is_empty() {
                        break;
                    }
                }

                self.set.commit(self.index, 0);
            }
        }
    }
}

struct ChunkReader {
    excess_len: usize,
    excess: [u8; CHUNK_ALIGN],
}

impl ChunkReader {
    pub fn new() -> Self {
        Self {
            excess_len: 0,
            excess: [0u8; CHUNK_ALIGN],
        }
    }
}
impl ChunkReader {
    #[inline(always)]
    pub fn read_chunk<const S: usize>(&mut self, reader: &mut impl Read, out: &mut [u8]) -> usize {
        // Copy excess from a previous read to the start of the buffer
        out[..self.excess_len].copy_from_slice(&self.excess[..self.excess_len]);

        // prev_excess_len is the length of the excess currently sitting at the start of the buffer
        let prev_excess_len = self.excess_len;

        // Read new data after excess
        // Todo: check if the read chunk size can be made exact (e.g not depending on excess_len)
        let mut nread_total = 0;

        loop {
            let nread = reader
                .read(&mut out[prev_excess_len + nread_total..])
                .unwrap();

            nread_total += nread;

            // Calculate new newline cutoff for the current chunk
            self.excess_len = if let Some(excess_len) = out
                .iter()
                .skip(prev_excess_len)
                .take(nread_total)
                .rev()
                .position(|&c| c == b'\n')
            {
                excess_len
            } else {
                0
            };

            if self.excess_len > 0 || nread == 0 {
                break;
            }
        }

        // Copy new excess to the excess buffer
        let excess_start = prev_excess_len + nread_total - self.excess_len;
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

#[target_feature(enable = "avx512f,sse4.2")]
unsafe fn process_stations<'a>(
    measurements_filepath: &str,
    station_map: &'a mut StationMap,
) -> Vec<(String, &'a StationAggregate)> {
    let f = File::open(measurements_filepath).unwrap();
    let mut file_reader = BufReader::new(f);
    let mut chunk_reader = ChunkReader::new();

    let ws = WorkSet::<NUM_WORKERS, Batch>::new();

    let results = std::thread::scope(|s| {
        let ws = &ws;

        let worker_threads = (0..NUM_WORKERS)
            .map(|ii| {
                let mut worker = Worker::new(ii, ws);

                s.spawn(move || {
                    unsafe {
                        worker.run();
                    }

                    return worker.map;
                })
            })
            .collect::<Vec<_>>();

        loop {
            let (index, set) = ws.acquire();

            let nread = chunk_reader
                .read_chunk::<CHUNK_SIZE>(&mut file_reader, &mut set.chunk[..CHUNK_SIZE]);

            if nread == 0 {
                break;
            }

            ws.commit(index, nread.try_into().unwrap());
        }

        ws.close();

        let results = worker_threads
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
                let s = _mm512_loadu_si512(slot.name.as_ptr() as *const _);
                let hash = hash_512!(s);

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
    use std::io::Write;

    let start_time = std::time::Instant::now();
    const FILE_PATH: &str = "data/measurements.txt";

    let mut station_map = StationMap {
        map: fastmap::FastMap::new(),
    };

    let stations = unsafe { process_stations(FILE_PATH, &mut station_map) };

    let print_start = std::time::Instant::now();

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
    buf_writer.write(b"}").unwrap();

    buf_writer.flush().unwrap();

    let print_time = print_start.elapsed();

    println!("\nPrint took: {}ms", print_time.as_millis());

    println!("Took: {}ms", start_time.elapsed().as_millis());

    println!("Done");
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        fs,
        hash::{DefaultHasher, Hash, Hasher},
        io::BufRead,
    };

    use super::*;

    const SAMPLE_PATH: &str = "data/sample16kb.txt";
    const STATIONS_PATH: &str = "data/weather_stations.csv";

    fn get_hash(vec: Vec<u8>) -> u64 {
        let mut hash = DefaultHasher::new();
        vec.hash(&mut hash);
        hash.finish()
    }

    fn begin_sample() -> (ChunkReader, BufReader<File>, u64) {
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
            let read_size = chunk_reader.read_chunk::<S>(&mut file_reader, &mut chunk);

            reconstructed.extend_from_slice(&chunk[..read_size]);

            if read_size == 0 {
                break;
            }
        }

        assert_eq!(get_hash(reconstructed), sample_hash, "hash mismatch");
    }

    fn naive(path: &str) -> BTreeMap<String, StationAggregate> {
        let mut index: BTreeMap<String, StationAggregate> = BTreeMap::new();

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

                        let mut name_bytes = [0u8; 64];

                        for i in 0..name_len {
                            name_bytes[i] = buf.as_bytes()[i];
                        }

                        let temp = buf[name_len + 1..].trim().parse::<f64>().unwrap();

                        let name = buf[..name_len].to_string();

                        let entry = index.entry(name).or_insert(StationAggregate::new());

                        entry.name = name_bytes;
                        entry.add((temp * 10.0) as i16);

                        break 'top;
                    }
                    _ => {}
                }
            }
        }

        index
    }

    #[test]
    fn test_chunked_reading() {
        reconstruct_test::<10>();
        reconstruct_test::<32>();
        reconstruct_test::<64>();
        reconstruct_test::<512>();
        reconstruct_test::<1024>();
        reconstruct_test::<4096>();
    }

    #[test]
    fn test_process_chunk_correctness() {
        let naive_result = naive(SAMPLE_PATH);

        let mut station_map = StationMap {
            map: fastmap::FastMap::new(),
        };

        let stations = unsafe { process_stations(SAMPLE_PATH, &mut station_map) };

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
            let mut string_bytes = [0u8; 64];
            for (i, byte) in station_name.bytes().enumerate() {
                string_bytes[i] = byte;
            }
            let h = unsafe {
                let s = _mm512_loadu_si512(string_bytes.as_ptr() as *const _);
                let h = hash_512!(s);
                StationMap::hash_u32(h)
            };

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

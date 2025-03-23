AVX-512 accelerated Rust implementation for processing a [1 billion row file](https://github.com/gunnarmorling/1brc) that contains weather readings.

#### Running locally

1. Follow the instructions in the official [1brc repository](https://github.com/gunnarmorling/1brc?tab=readme-ov-file#running-the-challenge) to create the challenge file into `data/measurements.txt`
2. Configure rust to use nightly `rustup override set nightly`
3. Run with `cargo run -r`. **Note that the project requires the following CPU features:** `avx,avx2,sse2,sse3,sse4.2,avx512f,avx512bw,avx512vl,avx512cd,avx512vbmi,avx512vbmi2,bmi1,popcnt`.
   - Missing any of the extension will make the program not run correctly.

#### Design & optimisations

The program reads measurements.txt from the disk in chunks, aligns them on a 64-byte boundary and zeroes out any data read past the last line, appending it to the start of the subsequent chunk. The chunk is then released for a worker thread to be processed. Processing happens with avx-512 where each line of the chunk is processed in a single, branchless block using vector instructions. Station name is parsed from the input and hashed into a custom hashmap structure that resolves a slot for the given station in constant time without branching. The temperature is also parsed and converted into integer format. Finally after the whole file has been read and processed, results from each worker are aggregated and printed to stdout.

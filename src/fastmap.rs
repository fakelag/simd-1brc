use std::arch::x86_64::*;

macro_rules! backing_size {
    () => {
        (1 << BUCKET_BITS) * (1 << SLOT_BITS)
    };
}

/// FastMap is a specialised branchless hash map that uses avx-512 instructions for fast lookups.
/// Resolving key collisions is a O(1) constant time operation where all keys in the same bucket
/// are compared with SIMD instructions at the same time, and an offset (slot) is calculated to find
/// the correct entry. The purpose of this implementation is to provide a small (cache-friendly) and
/// fast (branchless, SIMD accelerated) hash map provided that the user can provide a good hashing
/// function for their 32-bit integer keys that guarantees a sufficiently low collision rate
/// (see constraints below).
///
/// The following CPU extensions are required for this implementation:
/// `sse2,avx512f,avx512vl,bmi1,popcnt`
///
/// Constraints:
/// - `BUCKET_BITS` & `SLOT_BITS` must be positive, non-zero integers
/// - `SLOT_BITS` must be less than or equal to 4 (512/32 = max 16 slots per bucket)
/// - The user provided hash function to generate a `bucket`-index for a given set of keys to be stored
/// in this hash map must guarantee less than or equal to `(1<<SLOT_BITS)` equal `bucket`-index values
///
pub struct FastMap<const BUCKET_BITS: usize, const SLOT_BITS: usize, T>
where
    [(); backing_size!()]:,
{
    pub backing: Box<[T; backing_size!()]>,
    index: Box<[(u16, [u32; 1 << SLOT_BITS]); 1 << BUCKET_BITS]>,
}

impl<const BUCKET_BITS: usize, const SLOT_BITS: usize, T> FastMap<BUCKET_BITS, SLOT_BITS, T>
where
    T: Default + Copy + std::fmt::Debug,
    [u32; 1 << SLOT_BITS]: Default,
    [(); backing_size!()]:,
{
    const BACKING_SIZE: u32 = backing_size!();
    const INDEX_SIZE: u32 = 1 << BUCKET_BITS;
    const BUCKET_SIZE: u32 = 1 << SLOT_BITS;

    const BACKING_MASK: u32 = Self::BACKING_SIZE - 1;
    const INDEX_MASK: u32 = Self::INDEX_SIZE - 1;
    const SLOT_MASK: u32 = (1 << SLOT_BITS) - 1;

    pub fn new() -> Self {
        Self {
            backing: vec![T::default(); Self::BACKING_SIZE as usize]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            index: vec![Default::default(); 1 << BUCKET_BITS]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
        }
    }

    #[inline]
    #[target_feature(enable = "sse2,avx512f,avx512vl,bmi1,popcnt")]
    pub unsafe fn get_insert(&mut self, key: u32, bucket: u32) -> &mut T {
        debug_assert!(bucket < Self::INDEX_SIZE);

        unsafe {
            // Load index of the bucket. The first element is a mask of already populated slots.
            // The element contains an array of the populated keys. Next we will find which slot
            #[cfg(feature = "safety_checks")]
            let index = &mut self.index[bucket as usize];
            #[cfg(not(feature = "safety_checks"))]
            let index = &mut self.index.get_unchecked_mut(bucket as usize);

            // Load 512 bits from the index key-part into a vector. Depending on SLOT_BITS this
            // may load partially past the end of the array which won't be compared against due to the mask
            let index_vec = _mm512_loadu_epi32(index.1.as_ptr() as *const _);

            // Load key into 128-bit vector for collision detection
            let key_vec = _mm512_set1_epi32(key as i32);

            // Compare key against key vector in index using index.0 as mask
            let cmp_bits = _mm512_mask_cmp_epi32_mask(index.0, index_vec, key_vec, _MM_CMPINT_EQ);

            // Count how many bits are set in the mask, e.g the number of currently populated slots
            let insert_slot = index.0.count_ones();

            // Find a matching slot if any was found during the comparison above.
            // If no slots were found, _mm_tzcnt_32 will return 32 which is larger than
            // insert_slot could ever be (it has a max of 16)
            let match_slot = _mm_tzcnt_32(cmp_bits as u32) as u32;

            // Find slot for given key and bucket. Match slot will be 32 if no
            // match was found which will automatically select insert_slot
            let slot = match_slot.min(insert_slot as u32);

            // 1 if the bucket does not contain the key (an insert is needed)
            let is_miss_bit = (cmp_bits == 0) as u16;

            #[cfg(feature = "safety_checks")]
            {
                assert!(slot < Self::BUCKET_SIZE);
                assert!(bucket * Self::BUCKET_SIZE + slot < Self::BACKING_SIZE);
            }

            // Logic insert into the index that will not make
            // any changes if the slot is already populated
            *index.1.get_unchecked_mut(slot as usize) = key;
            index.0 <<= is_miss_bit;
            index.0 |= is_miss_bit;

            #[cfg(feature = "safety_checks")]
            return &mut self.backing[(bucket * Self::BUCKET_SIZE + slot) as usize];
            #[cfg(not(feature = "safety_checks"))]
            return self
                .backing
                .get_unchecked_mut((bucket * Self::BUCKET_SIZE + slot) as usize);
        }
    }

    pub const fn alloc_size() -> usize {
        std::mem::size_of::<T>() * Self::BACKING_SIZE as usize
            + std::mem::size_of::<(u16, [u32; 1 << SLOT_BITS])>() * (1 << BUCKET_BITS) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_same_bucket() {
        let mut map = FastMap::<8, 2, u32>::new();

        unsafe {
            for i in 0..4 {
                let ptr = map.get_insert(i, 0);
                *ptr = i;
            }

            for i in 0..4 {
                let ptr = map.get_insert(i, 0);
                assert_eq!(*ptr, i);
            }
        }
    }

    #[test]
    fn test_insert_mixed() {
        const BIT_SIZE: usize = 4;
        let mut map = FastMap::<BIT_SIZE, 2, u32>::new();

        fn hash(key: u8) -> u8 {
            0x88u8.wrapping_mul(key) >> 4
        }

        for i in 0..64 {
            let k = hash(i) as u32;

            unsafe {
                let ptr = map.get_insert(i as u32, k);
                *ptr = i as u32;
            }

            unsafe {
                let ptr = map.get_insert(i as u32, k);
                assert_eq!(*ptr, i as u32);
            }
        }

        for i in 0..64 {
            let k = hash(i) as u32;

            unsafe {
                let ptr = map.get_insert(i as u32, k);
                assert_eq!(*ptr, i as u32);
            }
        }
    }
}

// use std::{
//     cell::UnsafeCell,
//     mem::MaybeUninit,
//     sync::atomic::{AtomicU64, Ordering},
// };
use core::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicU64, Ordering, fence},
};

/// FastQueue is a ticket locked atomic queue with fixed size. Both push
/// and pop methods are blocking and wait indefinitely until a slot is available
pub struct FastQueue<const SET_SIZE: usize, T> {
    queue: Box<[(AtomicU64, MaybeUninit<UnsafeCell<T>>); SET_SIZE]>,
    head: AtomicU64,
    tail: AtomicU64,
}

unsafe impl<const SET_SIZE: usize, T: Send> Send for FastQueue<SET_SIZE, T> {}
unsafe impl<const SET_SIZE: usize, T: Send> Sync for FastQueue<SET_SIZE, T> {}

impl<const SET_SIZE: usize, T> FastQueue<SET_SIZE, T> {
    pub fn new() -> Self {
        FastQueue {
            queue: (0..SET_SIZE)
                .map(|_| (AtomicU64::new(0), MaybeUninit::uninit()))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
        }
    }

    pub fn push(&self, value: T) {
        // Acquire a new slot by incrementing tail
        let tail = self.tail.fetch_add(1, Ordering::Relaxed);

        // Idx points to the slot in the queue, ver is our "turn" to write to it.
        // ver is shifted right by one to reserve the least significant bit for the state:
        // (0=empty, can be written, 1=full, must be popped first)
        let idx = tail % SET_SIZE as u64;
        let ver = (tail / SET_SIZE as u64) << 1;

        // Wait for the slot to be empty. This can be relaxed because of the Acquire fence below
        let slot = &self.queue[idx as usize];
        while slot.0.load(Ordering::Relaxed) != ver {
            std::hint::spin_loop();
        }
        fence(Ordering::Acquire);

        unsafe {
            slot.1.assume_init_ref().get().write(value);
        }

        // Release the slot by setting the bottom bit of of ver to 1 to signal its full
        slot.0.store(ver + 1, Ordering::Release);
    }

    pub fn pop(&self) -> T {
        // Acquire a slot by incrementing head
        let head = self.head.fetch_add(1, Ordering::Relaxed);

        // Calculate idx & ver. See comment in push() for details
        let idx = head % SET_SIZE as u64;
        let ver = (head / SET_SIZE as u64) << 1;

        // Wait for the slot to be full. Acquire semantics below ensure that the slot can be popped
        let slot = &self.queue[idx as usize];
        while slot.0.load(Ordering::Relaxed) != ver + 1 {
            std::hint::spin_loop();
        }
        fence(Ordering::Acquire);

        let value = unsafe { slot.1.assume_init_read() };

        slot.0.store(ver + 2, Ordering::Release);

        value.into_inner()
    }
}

impl<const SET_SIZE: usize, T> Drop for FastQueue<SET_SIZE, T> {
    fn drop(&mut self) {
        loop {
            let head = self.head.fetch_add(1, Ordering::Relaxed);

            let idx = head % SET_SIZE as u64;
            let ver = (head / SET_SIZE as u64) << 1;

            let slot = &self.queue[idx as usize];
            if slot.0.load(Ordering::Acquire) != (ver | 1) {
                break;
            }

            let value = unsafe { slot.1.assume_init_read() };
            slot.0.store(ver + 2, Ordering::Release);

            let _ = value.into_inner();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn test_fastqueue() {
        const WORKERS: usize = 10;
        const ITERATIONS: usize = 10000;
        let queue = FastQueue::<5, Option<u64>>::new();

        let output = Arc::new(Mutex::new(Vec::new()));

        std::thread::scope(|s| {
            let queue = &queue;

            let _ = (0..WORKERS)
                .map(|_i| {
                    let output = output.clone();
                    s.spawn(move || {
                        loop {
                            if let Some(val) = queue.pop() {
                                output.lock().unwrap().push(val);
                            } else {
                                break;
                            }
                        }
                    })
                })
                .collect::<Vec<_>>();

            s.spawn(move || {
                for i in 0..ITERATIONS {
                    queue.push(Some(i as u64));
                }
                (0..WORKERS).for_each(|_| {
                    queue.push(None);
                });
            });
        });

        let worker_sum = output.lock().unwrap().iter().map(|v| *v).sum::<u64>();
        let producer_sum = (0..ITERATIONS).sum::<usize>() as u64;

        assert_eq!(
            worker_sum, producer_sum,
            "Worker sum: {}, Producer sum: {}",
            worker_sum, producer_sum
        );
    }

    #[test]
    fn test_fastqueue_overflow() {
        const ITERATIONS: u64 = 10;
        let queue = FastQueue::<5, Option<u64>>::new();

        let mut output = 0u64;

        std::thread::scope(|s| {
            let queue = &queue;

            for i in 0..ITERATIONS {
                s.spawn(move || {
                    queue.push(Some(i));
                });
            }

            while queue.tail.load(Ordering::Relaxed) != ITERATIONS {
                std::hint::spin_loop();
            }

            for _ in 0..ITERATIONS {
                output += queue.pop().unwrap();
            }

            // Queue should be empty
            assert!(queue.head.load(Ordering::Acquire) == queue.tail.load(Ordering::Acquire));
        });

        let expect = (0..ITERATIONS).sum::<u64>();

        assert_eq!(output, expect, "Expected {}, got {}", expect, output);
    }

    #[test]
    fn test_fastqueue_drop() {
        let counter = Arc::new(AtomicU64::new(0));
        #[derive(Debug)]
        struct TestType(u64, Arc<AtomicU64>);

        impl Drop for TestType {
            fn drop(&mut self) {
                assert_eq!(self.0, 1337, "Expected 1337, got {}", self.0);
                self.1.fetch_add(1, Ordering::SeqCst);
            }
        }

        {
            let queue = FastQueue::<4, TestType>::new();
            queue.push(TestType(1337, counter.clone()));
            queue.push(TestType(1337, counter.clone()));
            queue.pop(); // First drop from pop()
            // Second drop from queue drop
        }

        let dropcount = counter.load(Ordering::SeqCst);
        assert_eq!(dropcount, 2, "Expected 2, got {}", dropcount,);
    }

    // #[test]
    // fn test_fastqueue_bench() {
    //     use crossbeam_channel::bounded;

    //     const WORKERS: usize = 8;
    //     const ITERATIONS: usize = 10_000_000;
    //     const QUEUE_SIZE: usize = 1024;

    //     let (cb_send, cb_recv) = bounded::<usize>(QUEUE_SIZE);

    //     let rdtsc = unsafe { std::arch::x86_64::_rdtsc() };
    //     std::thread::scope(|s| {
    //         for _ in 0..WORKERS {
    //             let cb_recv = &cb_recv;
    //             let cb_send = &cb_send;

    //             s.spawn(move || {
    //                 for _ in 0..ITERATIONS {
    //                     let v = cb_recv.recv().unwrap();
    //                     std::hint::black_box(v);
    //                 }
    //             });

    //             s.spawn(move || {
    //                 for i in 0..ITERATIONS {
    //                     cb_send.send(i).unwrap();
    //                 }
    //             });
    //         }
    //     });
    //     let rdtsc_end = unsafe { std::arch::x86_64::_rdtsc() };
    //     let elapsed = rdtsc_end - rdtsc;

    //     let cycles_per_op = elapsed as f64 / (WORKERS * ITERATIONS) as f64;

    //     println!("{} cb cycles/op", cycles_per_op);

    //     let queue = FastQueue::<QUEUE_SIZE, usize>::new();

    //     let rdtsc = unsafe { std::arch::x86_64::_rdtsc() };
    //     std::thread::scope(|s| {
    //         let queue = &queue;
    //         for _ in 0..WORKERS {
    //             s.spawn(move || {
    //                 for _ in 0..ITERATIONS {
    //                     let v = queue.pop();
    //                     std::hint::black_box(v);
    //                 }
    //             });

    //             s.spawn(move || {
    //                 for i in 0..ITERATIONS {
    //                     queue.push(i);
    //                 }
    //             });
    //         }
    //     });
    //     let rdtsc_end = unsafe { std::arch::x86_64::_rdtsc() };
    //     let elapsed = rdtsc_end - rdtsc;

    //     let cycles_per_op = elapsed as f64 / (WORKERS * ITERATIONS) as f64;

    //     println!("{} fq cycles/op", cycles_per_op);
    // }
}

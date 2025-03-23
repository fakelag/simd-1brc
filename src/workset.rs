use std::{
    cell::UnsafeCell,
    sync::atomic::{AtomicU32, Ordering},
};

pub struct WorkSet<const SET_SIZE: usize, T> {
    pub set: [UnsafeCell<T>; SET_SIZE],
    signal: [AtomicU32; SET_SIZE],
}

unsafe impl<const SET_SIZE: usize, T: Send> Send for WorkSet<SET_SIZE, T> {}
unsafe impl<const SET_SIZE: usize, T: Send> Sync for WorkSet<SET_SIZE, T> {}

impl<const SET_SIZE: usize, T> WorkSet<SET_SIZE, T>
where
    T: Default,
{
    pub fn new() -> Self {
        WorkSet {
            set: (0..SET_SIZE)
                .map(|_| UnsafeCell::new(Default::default()))
                .collect::<Vec<UnsafeCell<T>>>()
                .try_into()
                .unwrap(),
            signal: (0..SET_SIZE)
                .map(|_| AtomicU32::new(0))
                .collect::<Vec<AtomicU32>>()
                .try_into()
                .unwrap(),
        }
    }

    /// Acquires the next free set. A set is considered free if its signal is 0
    #[inline(always)]
    pub fn acquire(&self) -> (usize, &mut T) {
        let index = 'top: loop {
            for (ii, sig) in self.signal.iter().enumerate() {
                if sig.load(Ordering::Relaxed) == 0 {
                    break 'top ii;
                }
            }
            std::hint::spin_loop();
        };

        std::sync::atomic::fence(Ordering::Acquire);

        unsafe { (index, &mut *UnsafeCell::raw_get(&self.set[index])) }
    }

    /// Closes all sets
    #[inline(always)]
    pub fn close(&self) {
        for sig in self.signal.iter() {
            while sig
                .compare_exchange(0, u32::MAX, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                std::thread::yield_now();
            }
        }
    }

    /// Commits a given set index with a non-zero signal. The signal s must satisfy `s < u32::MAX`
    #[inline(always)]
    pub fn commit(&self, index: usize, signal: u32) {
        debug_assert!(signal < u32::MAX);
        self.signal[index].store(signal, Ordering::Release);
    }

    /// Start work from a worker on its set. Returns `None` if the set is closed
    #[inline(always)]
    pub fn start(&self, index: usize) -> Option<(u32, &mut T)> {
        let signal = loop {
            let signal = self.signal[index].load(Ordering::Relaxed);

            if signal == u32::MAX {
                return None;
            } else if signal != 0 {
                break signal;
            }
        };

        std::sync::atomic::fence(Ordering::Acquire);

        unsafe { Some((signal, &mut *UnsafeCell::raw_get(&self.set[index]))) }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn test_workset() {
        const WORKERS: usize = 8;
        const ITERATIONS: usize = 10_000;
        let ws = WorkSet::<WORKERS, u32>::new();

        let output = Arc::new(Mutex::new(Vec::new()));

        std::thread::scope(|s| {
            let ws = &ws;
            // println!("Starting workers");

            let _ = (0..WORKERS)
                .map(|i| {
                    let output = output.clone();
                    s.spawn(move || {
                        loop {
                            if let Some((signal, set)) = ws.start(i) {
                                output.lock().unwrap().push((signal, *set));
                                ws.commit(i, 0);
                            } else {
                                // println!("Worker {} done", i);
                                break;
                            }
                        }
                    })
                })
                .collect::<Vec<_>>();

            s.spawn(move || {
                for i in 0..ITERATIONS {
                    let (index, set) = ws.acquire();
                    *set = i.try_into().unwrap();
                    ws.commit(index, 1);
                }
                // println!("Producer signaling workers");
                ws.close();
                // println!("Producer done");
            });
        });

        let worker_sum = output.lock().unwrap().iter().map(|(_, v)| *v).sum::<u32>();
        let producer_sum: u32 = (0..ITERATIONS).sum::<usize>() as u32;

        assert_eq!(
            worker_sum, producer_sum,
            "Worker sum: {}, Producer sum: {}",
            worker_sum, producer_sum
        );
    }
}

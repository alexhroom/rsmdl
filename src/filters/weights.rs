/// Class representation of an array of binary weights as a string of bits, 
/// stored in 64-bit chunks. 
///
/// This is used as an efficient way to represent a filter; in the histogram code,
/// a weight of 1 for an event indicates that an event should be included in the histogram,
/// whereas a weight of 0 means it should not be included.
use std::ops::{Index, BitAnd, Not};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator, IndexedParallelIterator};

/// Struct to store weights as a bit string.
/// The bits are stored in 64-bit chunks.
pub struct Weights
where
{
    // Note that each chunk of raw weights is stored in little-endian order;
    // that is, e.g. for the second chunk (representing weights 64-127)
    // the weight for value 64 would be the rightmost binary digit
    raw_weights: Vec<u64>,
    offset: usize
}

impl Weights {
    /// Create an array of all ones weights.
    pub fn ones(len: usize) -> Self {
        Weights {
            raw_weights: vec![u64::MAX; len / 64 + 1], offset: 0
        }
    }

    /// Create an array of all zero weights.
    pub fn zeros(len: usize) -> Self {
        Weights {
            raw_weights: vec![0; len / 64 + 1], offset: 0
        }
    }

    /// Set the weight at a given index to a given value.
    pub fn set_weight(&mut self, index: usize, set_to: bool) {
        match set_to {
            true => {
                self.raw_weights[index/64] |= 1 << (index % 64)
            }
            false => {
                self.raw_weights[index/64] &= !(1 << (index % 64))
            }
        }
    }

    /// Set a range of weights to a given value.
    pub fn set_range(&mut self, start: usize, end: usize, set_to: bool) {
        // round start up to the nearest chunk 
        let first_byte = match start % 64 {
            0 => start,
            _ => start + (63 - start % 64)
        };
        // round end down to the nearest chunk 
        let last_byte = match end % 64 {
            0 => end,
            _ => end - (start % 64)
        };

        // set bits individually where we aren't setting the full chunk
        for index in start..first_byte {
            self.set_weight(index, set_to);
        }
        for index in last_byte..end {
            self.set_weight(index, set_to);
        }

        // get value to set full bytes to
        let value = match set_to {
                true => u64::MAX,
                false => 0,
            };
        for byte in first_byte..last_byte {
            self.raw_weights[byte/64] = value
        }
    }

    /// Get an interval of weights between indices `start` and `end`.
    pub fn slice(&self, start: usize, end: usize) -> Weights {

        // round start down to the nearest chunk 
        let first_byte = match start % 64 {
            0 => start,
            _ => start - (start % 64)
        };
        // round end up to the nearest chunk 
        let last_byte = match end % 64 {
            0 => end,
            _ => end + (63 - start % 64)
        };

        // we take the full chunks that contain the given range and use the offset attribute
        // to handle the lower edge. note that overflow on the right hand side is possible,
        // but we don't iterate over these slices in the histogram code so doesn't happen
        Weights {
            raw_weights: self.raw_weights[first_byte/64..last_byte/64].to_vec(),
            offset: first_byte - start
        }
    }
}

// allow conversion of iterators of bools into Weights
impl<T: ExactSizeIterator> From<T> for Weights 
where
    T::Item: Into<bool>
{
    fn from(value: T) -> Self {
        let mut result = Weights::zeros(value.len());
        value 
            .into_iter()
            .enumerate()
            .for_each(|(k, v)| result.set_weight(k, v.into()));
        result
    }
}

// allow indexing
impl Index<usize> for Weights {
    type Output = bool;

    fn index(&self, index: usize) -> &bool {
        match (self.raw_weights[index/64 + self.offset] >> (index % 64)) & 1 {
            1 => &true,
            _ => &false,
        }
    }
}

impl BitAnd for Weights {
    type Output = Weights;

    fn bitand(self, rhs: Self) -> Self::Output {
        // we shouldn't ever need to combine slices, just full weight sets
        if (self.offset != 0) | (rhs.offset != 0) {
            panic!("Can only combine weights with no offset.")
        };

        // we simply iterate bitwise OR over the chunks
        Weights {
            raw_weights: self.raw_weights
                .par_iter()
                .zip(rhs.raw_weights.par_iter())
                .map(|(x, y)| x & y)
                .collect(),
            offset: 0
        }
    }
}

impl Not for Weights {
    type Output = Weights;

    fn not(self) -> Self::Output {
        Weights {
            raw_weights: self.raw_weights
                .par_iter()
                .map(|x| !x)
                .collect(),
            offset: self.offset
        } 
    }
}

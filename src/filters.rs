use std::ops::Index;

/// Struct to store weights as a bit string.
/// This provides convenience methods that hide
/// the internal bit string representation.
pub struct Weights {
    raw_weights: Vec<u8>,
    padding: usize
}

impl Weights {
    /// Create an array of all ones weights.
    pub fn ones(len: usize) -> Self {
        Weights {
            raw_weights: vec![255; len / 8 + 1], padding: 0
        }
    }

    /// Create an array of all zero weights.
    pub fn zeros(len: usize) -> Self {
        Weights {
            raw_weights: vec![0; len / 8 + 1], padding: 0
        }
    }

    /// Set the weight at a given index to a given value.
    pub fn set_weight(&mut self, index: usize, set_to: bool) {
        match set_to {
            true => {
                self.raw_weights[index/8] = self.raw_weights[index/8] as u8 | (1 << (7 - index % 8))
            }
            false => {
                self.raw_weights[index/8] = self.raw_weights[index/8] as u8 & !(1 << (7 - index % 8))
            }
        }
    }

    /// Get an interval of weights between indices `start` and `end`.
    pub fn slice(&self, start: usize, end: usize) -> Weights {

        // round start up to the nearest byte
        let first_byte = match start % 8 {
            0 => start,
            _ => start + (7 - start % 8)
        };
        // round end down to the nearest byte
        let last_byte = match end % 8 {
            0 => end,
            _ => end - (start % 8)
        };

        Weights {
            raw_weights: self.raw_weights[first_byte/8..last_byte/8].to_vec(),
            padding: first_byte - start
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
        match (self.raw_weights[index/8 + self.padding] >> (7 - (index % 8))) & 1 {
            1 => &true,
            0 => &false,
            _ => unreachable!()  // can never happen due to & 1
        }
    }
}

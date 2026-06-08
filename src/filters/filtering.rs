use crate::filters::weights::Weights;

/// Get a weights array corresponding to the filtered frames.
///
/// Parameters
/// ----------
/// f_start: Vec<usize>
///     The start frames indices for the filters.
/// f_end: Vec<usize>
///     The end frame indices for the filters.
/// start_index: Vec<usize>
///     A list of the first indices for each frame.
/// array_len: usize
///     The length of the final weights array.
///
/// Returns
/// -------
/// Weights
///     An array of the weights corresponding to the filtered frames.
///
pub fn get_good_values(
    f_start: Vec<usize>,
    f_end: Vec<usize>,
    start_index: Vec<usize>,
    array_len: usize,
) -> Weights {
    let mut result = Weights::zeros(array_len);

    f_start
        .into_iter()
        .zip(f_end.iter())
        .for_each(|(start, end)| {
            result.set_range(start_index[start], start_index[*end], true);
        });

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test the mask is created correctly for one filter.
    #[test]
    fn test_good_values_one_filter() {
        let f_start = vec![1];
        let f_end = vec![2];
        let start_index = vec![0, 30, 50, 64];
        let array_len = 64;

        let weights = get_good_values(f_start, f_end, start_index, array_len);

        // expected is 1s between index 30 and 50
        assert_eq!(weights, Weights::from_raw(vec![1125898833100800]))
    }

    /// Test the mask is created correctly for multiple filters.
    #[test]
    fn test_good_values_two_filters() {
        let f_start = vec![1, 4];
        let f_end = vec![2, 6];
        let start_index = vec![0, 10, 20, 30, 40, 50, 64];
        let array_len = 64;

        let weights = get_good_values(f_start, f_end, start_index, array_len);

        // expected is 1s between indices 10-20 and 40-64
        assert_eq!(weights, Weights::from_raw(vec![18446742974198971392]))
    }
}

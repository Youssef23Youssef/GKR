//! Multilinear extension utilities.
//!
//! This module provides the core polynomial utilities used by the circuit,
//! wiring, Sumcheck, and GKR layers. Evaluation tables use little-endian
//! Boolean indexing: the first variable is the least-significant bit of the
//! table index.
//!
//! For example, with two variables:
//!
//! ```text
//! index 0 -> [false, false]
//! index 1 -> [true,  false]
//! index 2 -> [false, true ]
//! index 3 -> [true,  true ]
//! ```
//!
//! This convention also determines variable binding: binding `x_0 = r` folds
//! adjacent pairs as `(1 - r) * old[2j] + r * old[2j + 1]`.

use crate::field::F;
use ark_ff::{One, Zero};

/// Errors produced by multilinear extension utilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MleError {
    /// Evaluation tables must contain at least one value.
    EmptyEvaluationTable,

    /// Evaluation table length must be a Boolean-hypercube size, `2^n`.
    EvaluationTableLengthNotPowerOfTwo { len: usize },

    /// An index is outside the Boolean hypercube for the requested bit length.
    IndexOutOfRange { index: usize, num_bits: usize },

    /// The requested number of variables cannot be represented safely by `usize`.
    TooManyVariables { num_bits: usize },

    /// A point has the wrong number of coordinates.
    PointLengthMismatch { expected: usize, actual: usize },

    /// A constant evaluation table has no variable left to bind.
    NoVariablesToBind,
}

/// Returns the number of Boolean variables represented by an evaluation table.
///
/// The table length must be a power of two. A length-one table represents a
/// constant polynomial with zero variables.
pub fn num_vars_for_len(len: usize) -> Result<usize, MleError> {
    if len == 0 {
        return Err(MleError::EmptyEvaluationTable);
    }

    if !len.is_power_of_two() {
        return Err(MleError::EvaluationTableLengthNotPowerOfTwo { len });
    }

    Ok(len.trailing_zeros() as usize)
}

/// Converts a table index into little-endian Boolean bits.
///
/// The returned vector has exactly `num_bits` entries.
pub fn index_to_bits(index: usize, num_bits: usize) -> Result<Vec<bool>, MleError> {
    if num_bits > usize::BITS as usize {
        return Err(MleError::TooManyVariables { num_bits });
    }

    if num_bits < usize::BITS as usize {
        let domain_size = 1usize << num_bits;

        if index >= domain_size {
            return Err(MleError::IndexOutOfRange { index, num_bits });
        }
    }

    let bits = (0..num_bits)
        .map(|bit_position| ((index >> bit_position) & 1) == 1)
        .collect();

    Ok(bits)
}

/// Converts little-endian Boolean bits into a table index.
pub fn bits_to_index(bits: &[bool]) -> Result<usize, MleError> {
    if bits.len() > usize::BITS as usize {
        return Err(MleError::TooManyVariables {
            num_bits: bits.len(),
        });
    }

    let mut index = 0usize;

    for (bit_position, bit) in bits.iter().enumerate() {
        if *bit {
            index |= 1usize << bit_position;
        }
    }

    Ok(index)
}

/// Evaluates the multilinear equality polynomial `eq(point, bits)`.
///
/// On Boolean inputs, this acts as an indicator: it is `1` when `point` equals
/// `bits`, and `0` otherwise.
pub fn eq(point: &[F], bits: &[bool]) -> Result<F, MleError> {
    if point.len() != bits.len() {
        return Err(MleError::PointLengthMismatch {
            expected: bits.len(),
            actual: point.len(),
        });
    }

    let one = F::one();

    let mut result = F::one();

    for (x_i, b_i) in point.iter().zip(bits.iter()) {
        let factor = if *b_i { *x_i } else { one - *x_i };

        result *= factor;
    }

    Ok(result)
}

/// Evaluates the multilinear extension represented by a Boolean-hypercube table.
///
/// `values` uses this module's little-endian indexing convention.
pub fn evaluate_mle(values: &[F], point: &[F]) -> Result<F, MleError> {
    let num_vars = num_vars_for_len(values.len())?;

    if point.len() != num_vars {
        return Err(MleError::PointLengthMismatch {
            expected: num_vars,
            actual: point.len(),
        });
    }

    let mut result = F::zero();

    for (index, value) in values.iter().enumerate() {
        let bits = index_to_bits(index, num_vars)?;
        let weight = eq(point, &bits)?;

        result += *value * weight;
    }

    Ok(result)
}

/// Binds the first variable `x_0` of an evaluation table to `r`.
///
/// With little-endian indexing, this folds adjacent pairs as
/// `(1 - r) * old[2j] + r * old[2j + 1]`.
pub fn bind_variable(values: &[F], r: F) -> Result<Vec<F>, MleError> {
    let num_vars = num_vars_for_len(values.len())?;

    if num_vars == 0 {
        return Err(MleError::NoVariablesToBind);
    }

    let one_minus_r = F::one() - r;
    let mut bound_values = Vec::with_capacity(values.len() / 2);

    for pair in values.chunks_exact(2) {
        let left = pair[0];
        let right = pair[1];

        bound_values.push(one_minus_r * left + r * right);
    }

    Ok(bound_values)
}

/// Evaluates the affine line from `p0` to `p1` at parameter `t`.
///
/// The line is `p0 + t * (p1 - p0)`, so `t = 0` gives `p0` and `t = 1` gives
/// `p1`.
pub fn affine_line(p0: &[F], p1: &[F], t: F) -> Result<Vec<F>, MleError> {
    if p0.len() != p1.len() {
        return Err(MleError::PointLengthMismatch {
            expected: p0.len(),
            actual: p1.len(),
        });
    }

    let point = p0
        .iter()
        .zip(p1.iter())
        .map(|(p0_i, p1_i)| *p0_i + t * (*p1_i - *p0_i))
        .collect();

    Ok(point)
}

/// Evaluates an MLE on the affine line from `p0` to `p1`.
///
/// This computes `Ṽ(p0 + t * (p1 - p0))`, the line restriction used later by
/// GKR to combine two child claims into one next-layer claim.
pub fn evaluate_mle_on_line(values: &[F], p0: &[F], p1: &[F], t: F) -> Result<F, MleError> {
    let point = affine_line(p0, p1, t)?;

    evaluate_mle(values, &point)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Indexing and table-shape tests

    #[test]
    fn num_vars_for_len_accepts_power_of_two_lengths() {
        assert_eq!(num_vars_for_len(1), Ok(0));
        assert_eq!(num_vars_for_len(2), Ok(1));
        assert_eq!(num_vars_for_len(4), Ok(2));
        assert_eq!(num_vars_for_len(8), Ok(3));
    }

    #[test]
    fn num_vars_for_len_rejects_zero_length() {
        assert_eq!(num_vars_for_len(0), Err(MleError::EmptyEvaluationTable));
    }

    #[test]
    fn num_vars_for_len_rejects_non_power_of_two_lengths() {
        assert_eq!(
            num_vars_for_len(3),
            Err(MleError::EvaluationTableLengthNotPowerOfTwo { len: 3 })
        );
        assert_eq!(
            num_vars_for_len(6),
            Err(MleError::EvaluationTableLengthNotPowerOfTwo { len: 6 })
        );
    }

    #[test]
    fn index_to_bits_handles_zero_variable_constant_case() {
        assert_eq!(index_to_bits(0, 0), Ok(vec![]));
    }

    #[test]
    fn index_to_bits_uses_little_endian_order() {
        assert_eq!(index_to_bits(0, 2), Ok(vec![false, false]));
        assert_eq!(index_to_bits(1, 2), Ok(vec![true, false]));
        assert_eq!(index_to_bits(2, 2), Ok(vec![false, true]));
        assert_eq!(index_to_bits(3, 2), Ok(vec![true, true]));
    }

    #[test]
    fn index_to_bits_rejects_index_outside_hypercube() {
        assert_eq!(
            index_to_bits(4, 2),
            Err(MleError::IndexOutOfRange {
                index: 4,
                num_bits: 2,
            })
        );
    }

    #[test]
    fn bits_to_index_uses_little_endian_order() {
        assert_eq!(bits_to_index(&[false, false]), Ok(0));
        assert_eq!(bits_to_index(&[true, false]), Ok(1));
        assert_eq!(bits_to_index(&[false, true]), Ok(2));
        assert_eq!(bits_to_index(&[true, true]), Ok(3));
    }

    #[test]
    fn bit_index_conversion_round_trips() {
        for num_bits in 0..6 {
            let domain_size = 1usize << num_bits;

            for index in 0..domain_size {
                let bits = index_to_bits(index, num_bits).unwrap();
                assert_eq!(bits_to_index(&bits), Ok(index));
            }
        }
    }

    // `eq()` tests

    #[test]
    fn eq_of_empty_points_is_one() {
        assert_eq!(eq(&[], &[]), Ok(F::one()));
    }

    #[test]
    fn eq_is_one_when_boolean_points_match() {
        let point = vec![F::zero(), F::one()];

        assert_eq!(eq(&point, &[false, true]), Ok(F::one()));
    }

    #[test]
    fn eq_is_zero_when_boolean_points_do_not_match() {
        let point = vec![F::one(), F::one()];

        assert_eq!(eq(&point, &[false, true]), Ok(F::zero()));
    }

    #[test]
    fn eq_uses_one_minus_x_for_false_bits() {
        let r = F::from(7u64);
        let s = F::from(11u64);

        let point = vec![r, s];

        assert_eq!(eq(&point, &[false, true]), Ok((F::one() - r) * s));
    }

    #[test]
    fn eq_rejects_length_mismatch() {
        let point = vec![F::from(7u64)];

        assert_eq!(
            eq(&point, &[true, false]),
            Err(MleError::PointLengthMismatch {
                expected: 2,
                actual: 1,
            })
        );
    }

    // Evaluating MLE

    #[test]
    fn evaluate_mle_handles_constant_table() {
        let values = vec![F::from(7u64)];
        let point = vec![];

        assert_eq!(evaluate_mle(&values, &point), Ok(F::from(7u64)));
    }

    #[test]
    fn evaluate_mle_matches_one_variable_boolean_points() {
        let values = vec![F::from(5u64), F::from(12u64)];

        assert_eq!(evaluate_mle(&values, &[F::zero()]), Ok(F::from(5u64)));
        assert_eq!(evaluate_mle(&values, &[F::one()]), Ok(F::from(12u64)));
    }

    #[test]
    fn evaluate_mle_interpolates_one_variable_table() {
        let values = vec![F::from(5u64), F::from(12u64)];
        let r = F::from(9u64);

        let expected = F::from(5u64) * (F::one() - r) + F::from(12u64) * r;

        assert_eq!(evaluate_mle(&values, &[r]), Ok(expected));
    }

    #[test]
    fn evaluate_mle_matches_two_variable_boolean_points() {
        let values = vec![F::from(2u64), F::from(3u64), F::from(5u64), F::from(7u64)];

        assert_eq!(
            evaluate_mle(&values, &[F::zero(), F::zero()]),
            Ok(F::from(2u64))
        );
        assert_eq!(
            evaluate_mle(&values, &[F::one(), F::zero()]),
            Ok(F::from(3u64))
        );
        assert_eq!(
            evaluate_mle(&values, &[F::zero(), F::one()]),
            Ok(F::from(5u64))
        );
        assert_eq!(
            evaluate_mle(&values, &[F::one(), F::one()]),
            Ok(F::from(7u64))
        );
    }

    #[test]
    fn evaluate_mle_rejects_point_length_mismatch() {
        let values = vec![F::from(2u64), F::from(3u64), F::from(5u64), F::from(7u64)];

        assert_eq!(
            evaluate_mle(&values, &[F::from(9u64)]),
            Err(MleError::PointLengthMismatch {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn evaluate_mle_rejects_non_power_of_two_table() {
        let values = vec![F::from(2u64), F::from(3u64), F::from(5u64)];
        let point = vec![F::from(9u64)];

        assert_eq!(
            evaluate_mle(&values, &point),
            Err(MleError::EvaluationTableLengthNotPowerOfTwo { len: 3 })
        );
    }

    // Bind variable tests
    #[test]
    fn bind_variable_folds_two_value_table() {
        let values = vec![F::from(5u64), F::from(12u64)];
        let r = F::from(9u64);

        let expected = vec![(F::one() - r) * F::from(5u64) + r * F::from(12u64)];

        assert_eq!(bind_variable(&values, r), Ok(expected));
    }

    #[test]
    fn bind_variable_at_zero_selects_even_indices() {
        let values = vec![F::from(2u64), F::from(3u64), F::from(5u64), F::from(7u64)];

        assert_eq!(
            bind_variable(&values, F::zero()),
            Ok(vec![F::from(2u64), F::from(5u64)])
        );
    }

    #[test]
    fn bind_variable_at_one_selects_odd_indices() {
        let values = vec![F::from(2u64), F::from(3u64), F::from(5u64), F::from(7u64)];

        assert_eq!(
            bind_variable(&values, F::one()),
            Ok(vec![F::from(3u64), F::from(7u64)])
        );
    }

    #[test]
    fn bind_variable_interpolates_adjacent_pairs() {
        let values = vec![F::from(2u64), F::from(3u64), F::from(5u64), F::from(7u64)];
        let r = F::from(11u64);

        let expected = vec![
            (F::one() - r) * F::from(2u64) + r * F::from(3u64),
            (F::one() - r) * F::from(5u64) + r * F::from(7u64),
        ];

        assert_eq!(bind_variable(&values, r), Ok(expected));
    }

    #[test]
    fn bind_variable_rejects_non_power_of_two_table() {
        let values = vec![F::from(2u64), F::from(3u64), F::from(5u64)];

        assert_eq!(
            bind_variable(&values, F::from(9u64)),
            Err(MleError::EvaluationTableLengthNotPowerOfTwo { len: 3 })
        );
    }

    #[test]
    fn bind_variable_rejects_constant_table() {
        let values = vec![F::from(7u64)];

        assert_eq!(
            bind_variable(&values, F::from(9u64)),
            Err(MleError::NoVariablesToBind)
        );
    }

    // Affine line tests
    #[test]
    fn affine_line_at_zero_returns_first_endpoint() {
        let p0 = vec![F::from(2u64), F::from(5u64)];
        let p1 = vec![F::from(8u64), F::from(17u64)];

        assert_eq!(affine_line(&p0, &p1, F::zero()), Ok(p0));
    }

    #[test]
    fn affine_line_at_one_returns_second_endpoint() {
        let p0 = vec![F::from(2u64), F::from(5u64)];
        let p1 = vec![F::from(8u64), F::from(17u64)];

        assert_eq!(affine_line(&p0, &p1, F::one()), Ok(p1));
    }

    #[test]
    fn affine_line_interpolates_coordinate_wise() {
        let p0 = vec![F::from(2u64), F::from(5u64)];
        let p1 = vec![F::from(8u64), F::from(17u64)];
        let t = F::from(3u64);

        let expected = vec![
            F::from(2u64) + t * (F::from(8u64) - F::from(2u64)),
            F::from(5u64) + t * (F::from(17u64) - F::from(5u64)),
        ];

        assert_eq!(affine_line(&p0, &p1, t), Ok(expected));
    }

    #[test]
    fn affine_line_rejects_endpoint_length_mismatch() {
        let p0 = vec![F::from(2u64), F::from(5u64)];
        let p1 = vec![F::from(8u64)];

        assert_eq!(
            affine_line(&p0, &p1, F::from(3u64)),
            Err(MleError::PointLengthMismatch {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn evaluate_mle_on_line_at_zero_matches_first_endpoint_evaluation() {
        let values = vec![F::from(2u64), F::from(3u64), F::from(5u64), F::from(7u64)];

        let p0 = vec![F::zero(), F::zero()];
        let p1 = vec![F::one(), F::one()];

        assert_eq!(
            evaluate_mle_on_line(&values, &p0, &p1, F::zero()),
            Ok(F::from(2u64))
        );
    }

    #[test]
    fn evaluate_mle_on_line_at_one_matches_second_endpoint_evaluation() {
        let values = vec![F::from(2u64), F::from(3u64), F::from(5u64), F::from(7u64)];

        let p0 = vec![F::zero(), F::zero()];
        let p1 = vec![F::one(), F::one()];

        assert_eq!(
            evaluate_mle_on_line(&values, &p0, &p1, F::one()),
            Ok(F::from(7u64))
        );
    }

    #[test]
    fn evaluate_mle_on_line_matches_evaluate_mle_at_line_point() {
        let values = vec![F::from(2u64), F::from(3u64), F::from(5u64), F::from(7u64)];

        let p0 = vec![F::zero(), F::zero()];
        let p1 = vec![F::one(), F::one()];
        let t = F::from(3u64);

        let line_point = affine_line(&p0, &p1, t).unwrap();

        assert_eq!(
            evaluate_mle_on_line(&values, &p0, &p1, t),
            evaluate_mle(&values, &line_point)
        );
    }
}

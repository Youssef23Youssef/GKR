use crate::field::F;
use ark_ff::{One, Zero};

// Multilinear-extension utilities.
//
// This module starts with the indexing convention that every later MLE,
// wiring-predicate, Sumcheck, and GKR function must agree on.
//
// We use **little-endian Boolean indexing**:
//
// ```text
// index 0 -> [false, false]  // 00₂
// index 1 -> [true,  false]  // 01₂
// index 2 -> [false, true ]  // 10₂
// index 3 -> [true,  true ]  // 11₂
// ```
//
// In other words, the first Boolean variable is the least-significant bit of
// the vector index. This matches the folding rule we will use later for MLEs:
//
// ```text
// bind x₀ = r:
// new[j] = (1 - r) * old[2j] + r * old[2j + 1]
//```
//
// Being explicit here matters because a mismatched bit order would make layer
// MLEs and wiring predicates silently disagree.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MleError {
    /// MLE evaluation tables must contain at least one value.
    EmptyEvaluationTable,

    /// MLE evaluation tables are indexed by Boolean hypercubes, so their length
    /// must be exactly `2^num_vars`.
    EvaluationTableLengthNotPowerOfTwo { len: usize },

    /// The requested vector index cannot be represented with the given number
    /// of Boolean bits.
    IndexOutOfRange { index: usize, num_bits: usize },

    /// The requested number of Boolean variables is too large to safely map
    /// into a `usize` index on this machine.
    TooManyVariables { num_bits: usize },

    PointLengthMismatch {
        expected: usize,
        actual: usize,
    },
}

/// Return the number of Boolean variables needed for an evaluation table.
///
/// A multilinear polynomial in `n` variables has one value for every Boolean
/// point in `{0,1}^n`, so its evaluation table must have length `2^n`.
///
/// Examples:
///
/// ```text
/// len = 1 -> 0 variables
/// len = 2 -> 1 variable
/// len = 4 -> 2 variables
/// len = 8 -> 3 variables
/// ```
///
/// The `len = 1` case is important: the output layer of the basic GKR circuit
/// has one value, so its layer MLE is a constant polynomial with zero variables.
pub fn num_vars_for_len(len: usize) -> Result<usize, MleError> {
    if len == 0 {
        return Err(MleError::EmptyEvaluationTable);
    }

    if !len.is_power_of_two() {
        return Err(MleError::EvaluationTableLengthNotPowerOfTwo { len });
    }

    Ok(len.trailing_zeros() as usize)
}

/// Convert a vector index into little-endian Boolean bits.
///
/// The returned vector always has exactly `num_bits` entries. Bit position `0`
/// corresponds to the least-significant bit of `index`.
///
/// Examples with `num_bits = 2`:
///
/// ```text
/// index 0 -> [false, false]
/// index 1 -> [true,  false]
/// index 2 -> [false, true ]
/// index 3 -> [true,  true ]
/// ```
///
/// `index` must be less than `2^num_bits`; otherwise the index is outside the
/// Boolean hypercube described by `num_bits` variables.
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

/// Convert little-endian Boolean bits back into a vector index.
///
/// This is the inverse of `index_to_bits` for valid inputs.
///
/// Examples:
///
/// ```text
/// [false, false] -> 0
/// [true,  false] -> 1
/// [false, true ] -> 2
/// [true,  true ] -> 3
/// ```
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

/// Evaluates the multilinear equality polynomial at `(point, bits)`.
///
/// The input `point` is a field point `x ∈ F^n`, while `bits` is a Boolean
/// hypercube point `b ∈ {0,1}^n`.
///
/// The polynomial is defined as:
///
/// ```text
/// eq(x, b) = ∏ᵢ (bᵢ xᵢ + (1 - bᵢ)(1 - xᵢ))
/// ```
///
/// Since each `bᵢ` is Boolean, each factor is either `xᵢ` when `bᵢ = 1`,
/// or `1 - xᵢ` when `bᵢ = 0`.
///
/// On Boolean inputs, `eq` acts as an indicator polynomial: it evaluates to
/// `1` when the two Boolean points are equal, and `0` otherwise.
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
        let factor = if *b_i {
            *x_i
        } else {
            one - *x_i
        };

        result *= factor;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers tests

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
}
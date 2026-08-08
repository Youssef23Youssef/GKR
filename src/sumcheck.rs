//! Data structures for the Sumcheck protocol.
//!
//! Sumcheck reduces a claim about a multivariate polynomial to a sequence of
//! claims about univariate round polynomials. This module currently defines the
//! proof containers only, prover and verifier logic will be added later.

use crate::{
    field::F,
    mle::{MleError, index_to_bits},
};
use ark_ff::{Field, Zero};

/// Errors produced by Sumcheck data-structure validation and protocol logic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SumcheckError {
    /// A univariate polynomial must contain at least one coefficient.
    EmptyPolynomial,

    /// A multivariate evaluation point has the wrong number of coordinates.
    PointLengthMismatch { expected: usize, actual: usize },

    /// Interpolation requires one x-coordinate for each y-coordinate.
    InterpolationLengthMismatch { xs: usize, ys: usize },

    /// Lagrange interpolation requires distinct x-coordinates.
    DuplicateInterpolationPoint { first: usize, second: usize },

    /// The verifier challenge list must contain exactly one challenge per variable.
    ChallengeLengthMismatch { expected: usize, actual: usize },

    /// A Sumcheck proof must contain exactly one round polynomial per variable.
    WrongNumberOfRounds { expected: usize, actual: usize },

    /// A round polynomial cannot be built after all variables have already been fixed.
    FixedPrefixTooLong {
        num_vars: usize,
        fixed_prefix_len: usize,
    },

    /// No variable remains for the next Sumcheck round.
    NoVariablesRemaining {
        num_vars: usize,
        fixed_prefix_len: usize,
    },

    /// A verifier received a round polynomial above the allowed degree bound.
    RoundPolynomialDegreeTooHigh {
        round: usize,
        degree: usize,
        degree_bound: usize,
    },

    /// A round polynomial failed the Sumcheck consistency check.
    RoundConsistencyFailed { round: usize },

    /// The proof's final point does not match the verifier challenges.
    FinalPointMismatch,

    /// The proof's claimed final evaluation does not match the polynomial oracle.
    FinalEvaluationMismatch,

    /// The final folded claim does not match the proof's final evaluation.
    ClaimMismatch,

    /// The polynomial evaluator failed while proving or verifying.
    PolynomialEvaluationFailed,

    /// Error produced by the underlying MLE utilities.
    MleError(MleError),
}

impl From<MleError> for SumcheckError {
    fn from(error: MleError) -> Self {
        Self::MleError(error)
    }
}

/// Prover-side callback for evaluating a multivariate polynomial at a point.
pub type PolynomialEvaluator<'a> = &'a dyn Fn(&[F]) -> Result<F, SumcheckError>;

/// Enumerates all Boolean assignments of length `num_vars` as field elements.
///
/// The assignments use the same little-endian indexing convention as `mle.rs`:
///
/// ```text
/// num_vars = 0 -> [[]]
/// num_vars = 1 -> [[0], [1]]
/// num_vars = 2 -> [[0,0], [1,0], [0,1], [1,1]]
/// ```
///
/// This helper is used by the naive Sumcheck prover when constructing a round
/// polynomial by explicitly summing over the remaining Boolean suffix variables.
pub fn boolean_suffixes(num_vars: usize) -> Result<Vec<Vec<F>>, SumcheckError> {
    if num_vars >= usize::BITS as usize {
        return Err(SumcheckError::MleError(MleError::TooManyVariables {
            num_bits: num_vars,
        }));
    }

    let domain_size = 1usize << num_vars;
    let mut suffixes = Vec::with_capacity(domain_size);

    for index in 0..domain_size {
        let bits = index_to_bits(index, num_vars)?;
        let suffix = bits
            .into_iter()
            .map(|bit| if bit { F::from(1u64) } else { F::zero() })
            .collect();

        suffixes.push(suffix);
    }

    Ok(suffixes)
}

/// Builds one Sumcheck round polynomial by explicitly summing over Boolean suffixes.
///
/// If `fixed_prefix = [r_0, ..., r_{j-1}]`, this constructs:
///
/// ```text
/// g_j(t) = Σ f(r_0, ..., r_{j-1}, t, remaining_boolean_variables)
/// ```
///
/// The polynomial is reconstructed by evaluating it at `t = 0, 1, ..., degree_bound`
/// and interpolating those points. This is intentionally naive and optimized for
/// clarity rather than prover performance.
pub fn build_round_polynomial(
    polynomial: PolynomialEvaluator<'_>,
    num_vars: usize,
    fixed_prefix: &[F],
    degree_bound: usize,
) -> Result<UnivariatePolynomial, SumcheckError> {
    if fixed_prefix.len() > num_vars {
        return Err(SumcheckError::FixedPrefixTooLong {
            num_vars,
            fixed_prefix_len: fixed_prefix.len(),
        });
    }

    if fixed_prefix.len() == num_vars {
        return Err(SumcheckError::NoVariablesRemaining {
            num_vars,
            fixed_prefix_len: fixed_prefix.len(),
        });
    }

    let remaining_vars = num_vars - fixed_prefix.len() - 1;
    let suffixes = boolean_suffixes(remaining_vars)?;

    let xs: Vec<F> = (0..=degree_bound)
        .map(|value| F::from(value as u64))
        .collect();

    let mut ys = Vec::with_capacity(xs.len());

    for t in &xs {
        let mut sum = F::zero();

        for suffix in &suffixes {
            let mut point = Vec::with_capacity(num_vars);
            point.extend_from_slice(fixed_prefix);
            point.push(*t);
            point.extend_from_slice(suffix);

            sum += polynomial(&point)?;
        }

        ys.push(sum);
    }

    interpolate_univariate(&xs, &ys)
}

/// Interpolates a univariate polynomial from point-value pairs.
///
/// The returned polynomial is represented in ascending coefficient order. This
/// uses naive Lagrange interpolation and is intended for clarity, not speed.
pub fn interpolate_univariate(xs: &[F], ys: &[F]) -> Result<UnivariatePolynomial, SumcheckError> {
    if xs.len() != ys.len() {
        return Err(SumcheckError::InterpolationLengthMismatch {
            xs: xs.len(),
            ys: ys.len(),
        });
    }

    if xs.is_empty() {
        return Err(SumcheckError::EmptyPolynomial);
    }

    for first in 0..xs.len() {
        for second in (first + 1)..xs.len() {
            if xs[first] == xs[second] {
                return Err(SumcheckError::DuplicateInterpolationPoint { first, second });
            }
        }
    }

    let mut result = vec![F::zero(); xs.len()];

    for i in 0..xs.len() {
        let mut basis = vec![F::from(1u64)];
        let mut denominator = F::from(1u64);

        for j in 0..xs.len() {
            if i == j {
                continue;
            }

            basis = multiply_by_linear_factor(&basis, xs[j]);
            denominator *= xs[i] - xs[j];
        }

        let scale = ys[i]
            * denominator
                .inverse()
                .expect("duplicate x-coordinates were rejected before interpolation");

        for (coefficient, basis_coefficient) in result.iter_mut().zip(basis) {
            *coefficient += basis_coefficient * scale;
        }
    }

    UnivariatePolynomial::new(result)
}

/// Multiplies `poly` by `(t - root)`.
fn multiply_by_linear_factor(poly: &[F], root: F) -> Vec<F> {
    let mut result = vec![F::zero(); poly.len() + 1];

    for (degree, coefficient) in poly.iter().enumerate() {
        result[degree] -= *coefficient * root;
        result[degree + 1] += coefficient;
    }

    result
}

/// A univariate polynomial represented by coefficients in ascending degree order.
///
/// The coefficient vector stores:
///
/// ```text
/// coeffs[0] + coeffs[1] * t + coeffs[2] * t^2 + ...
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnivariatePolynomial {
    coeffs: Vec<F>,
}

impl UnivariatePolynomial {
    /// Creates a polynomial from coefficients in ascending degree order.
    pub fn new(coeffs: Vec<F>) -> Result<Self, SumcheckError> {
        if coeffs.is_empty() {
            return Err(SumcheckError::EmptyPolynomial);
        }

        Ok(Self { coeffs })
    }

    /// Creates a constant polynomial.
    pub fn constant(value: F) -> Self {
        Self {
            coeffs: vec![value],
        }
    }

    /// Returns the coefficients in ascending degree order.
    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    /// Returns the polynomial degree after ignoring trailing zero coefficients.
    pub fn degree(&self) -> usize {
        self.coeffs
            .iter()
            .rposition(|coeff| !coeff.is_zero())
            .unwrap_or(0)
    }

    /// Evaluates the polynomial at `point` using Horner's method.
    pub fn evaluate(&self, point: F) -> F {
        self.coeffs
            .iter()
            .rev()
            .fold(F::zero(), |acc, coeff| acc * point + coeff)
    }
}

/// A Sumcheck proof transcript.
///
/// Each entry in `round_polynomials` is the prover's univariate polynomial for
/// one Sumcheck round. After all verifier challenges are sampled, `final_point`
/// stores the resulting multivariate point and `final_evaluation` stores the
/// claimed evaluation of the original polynomial at that point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckProof {
    round_polynomials: Vec<UnivariatePolynomial>,
    final_point: Vec<F>,
    final_evaluation: F,
}

impl SumcheckProof {
    /// Creates a Sumcheck proof container.
    pub fn new(
        round_polynomials: Vec<UnivariatePolynomial>,
        final_point: Vec<F>,
        final_evaluation: F,
    ) -> Self {
        Self {
            round_polynomials,
            final_point,
            final_evaluation,
        }
    }

    /// Returns all univariate round polynomials in order.
    pub fn round_polynomials(&self) -> &[UnivariatePolynomial] {
        &self.round_polynomials
    }

    /// Returns the final verifier challenge point.
    pub fn final_point(&self) -> &[F] {
        &self.final_point
    }

    /// Returns the claimed final evaluation at `final_point`.
    pub fn final_evaluation(&self) -> F {
        self.final_evaluation
    }

    /// Returns the number of Sumcheck rounds represented by this proof.
    pub fn num_rounds(&self) -> usize {
        self.round_polynomials.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(value: u64) -> F {
        F::from(value)
    }

    #[test]
    fn boolean_suffixes_handles_zero_variables() {
        assert_eq!(boolean_suffixes(0), Ok(vec![vec![]]));
    }

    #[test]
    fn boolean_suffixes_handles_one_variable() {
        assert_eq!(boolean_suffixes(1), Ok(vec![vec![f(0)], vec![f(1)]]));
    }

    #[test]
    fn boolean_suffixes_uses_little_endian_order_for_two_variables() {
        assert_eq!(
            boolean_suffixes(2),
            Ok(vec![
                vec![f(0), f(0)],
                vec![f(1), f(0)],
                vec![f(0), f(1)],
                vec![f(1), f(1)],
            ])
        );
    }

    #[test]
    fn boolean_suffixes_rejects_too_many_variables() {
        assert_eq!(
            boolean_suffixes(usize::BITS as usize),
            Err(SumcheckError::MleError(MleError::TooManyVariables {
                num_bits: usize::BITS as usize,
            }))
        );
    }

    #[test]
    fn build_round_polynomial_constructs_first_round_for_linear_polynomial() {
        let polynomial: PolynomialEvaluator<'_> =
            &|point: &[F]| Ok(point[0] + f(2) * point[1] + f(5));

        let round = build_round_polynomial(polynomial, 2, &[], 1).unwrap();

        // g_0(t) = f(t, 0) + f(t, 1)
        //        = (t + 5) + (t + 7)
        //        = 2t + 12
        assert_eq!(round.coeffs(), &[f(12), f(2)]);
        assert_eq!(round.evaluate(f(0)), f(12));
        assert_eq!(round.evaluate(f(1)), f(14));
    }

    #[test]
    fn build_round_polynomial_constructs_later_round_with_fixed_prefix() {
        let polynomial: PolynomialEvaluator<'_> =
            &|point: &[F]| Ok(point[0] + f(2) * point[1] + f(5));

        let round = build_round_polynomial(polynomial, 2, &[f(3)], 1).unwrap();

        // g_1(t) = f(3, t)
        //        = 3 + 2t + 5
        //        = 2t + 8
        assert_eq!(round.coeffs(), &[f(8), f(2)]);
        assert_eq!(round.evaluate(f(0)), f(8));
        assert_eq!(round.evaluate(f(1)), f(10));
    }

    #[test]
    fn build_round_polynomial_supports_quadratic_degree_bound() {
        let polynomial: PolynomialEvaluator<'_> = &|point: &[F]| Ok(point[0] * point[0] + f(3));

        let round = build_round_polynomial(polynomial, 1, &[], 2).unwrap();

        // g_0(t) = t^2 + 3
        assert_eq!(round.coeffs(), &[f(3), f(0), f(1)]);
        assert_eq!(round.evaluate(f(0)), f(3));
        assert_eq!(round.evaluate(f(1)), f(4));
        assert_eq!(round.evaluate(f(2)), f(7));
    }

    #[test]
    fn build_round_polynomial_rejects_prefix_longer_than_num_vars() {
        let polynomial: PolynomialEvaluator<'_> = &|point: &[F]| Ok(point.iter().copied().sum());

        assert_eq!(
            build_round_polynomial(polynomial, 1, &[f(3), f(4)], 1),
            Err(SumcheckError::FixedPrefixTooLong {
                num_vars: 1,
                fixed_prefix_len: 2,
            })
        );
    }

    #[test]
    fn build_round_polynomial_rejects_when_no_variables_remain() {
        let polynomial: PolynomialEvaluator<'_> = &|point: &[F]| Ok(point.iter().copied().sum());

        assert_eq!(
            build_round_polynomial(polynomial, 2, &[f(3), f(4)], 1),
            Err(SumcheckError::NoVariablesRemaining {
                num_vars: 2,
                fixed_prefix_len: 2,
            })
        );
    }

    #[test]
    fn build_round_polynomial_propagates_polynomial_errors() {
        let polynomial: PolynomialEvaluator<'_> = &|point: &[F]| {
            Err(SumcheckError::PointLengthMismatch {
                expected: 99,
                actual: point.len(),
            })
        };

        assert_eq!(
            build_round_polynomial(polynomial, 2, &[], 1),
            Err(SumcheckError::PointLengthMismatch {
                expected: 99,
                actual: 2,
            })
        );
    }

    #[test]
    fn polynomial_requires_at_least_one_coefficient() {
        assert_eq!(
            UnivariatePolynomial::new(vec![]),
            Err(SumcheckError::EmptyPolynomial)
        );
    }

    #[test]
    fn polynomial_stores_coefficients_in_ascending_degree_order() {
        let polynomial = UnivariatePolynomial::new(vec![f(2), f(3), f(5)]).unwrap();

        assert_eq!(polynomial.coeffs(), &[f(2), f(3), f(5)]);
    }

    #[test]
    fn polynomial_degree_allows_quadratic_rounds() {
        let polynomial = UnivariatePolynomial::new(vec![f(2), f(3), f(5)]).unwrap();

        assert_eq!(polynomial.degree(), 2);
    }

    #[test]
    fn polynomial_degree_ignores_trailing_zero_coefficients() {
        let polynomial = UnivariatePolynomial::new(vec![f(2), f(3), F::zero()]).unwrap();

        assert_eq!(polynomial.degree(), 1);
    }

    #[test]
    fn polynomial_evaluates_at_field_point() {
        let polynomial = UnivariatePolynomial::new(vec![f(2), f(3), f(5)]).unwrap();

        assert_eq!(polynomial.evaluate(f(7)), f(268));
    }

    #[test]
    fn constant_polynomial_has_degree_zero() {
        let polynomial = UnivariatePolynomial::constant(f(9));

        assert_eq!(polynomial.degree(), 0);
        assert_eq!(polynomial.evaluate(f(123)), f(9));
    }

    #[test]
    fn polynomial_evaluator_closure_evaluates_multivariate_points() {
        let polynomial: PolynomialEvaluator<'_> = &|point: &[F]| {
            Ok(point
                .iter()
                .copied()
                .fold(F::zero(), |acc, value| acc + value))
        };

        assert_eq!(polynomial(&[f(2), f(3), f(5)]), Ok(f(10)));
    }

    #[test]
    fn polynomial_evaluator_closure_can_return_sumcheck_errors() {
        let polynomial: PolynomialEvaluator<'_> = &|point: &[F]| {
            if point.len() != 2 {
                return Err(SumcheckError::PointLengthMismatch {
                    expected: 2,
                    actual: point.len(),
                });
            }

            Ok(point[0] * point[1])
        };

        assert_eq!(polynomial(&[f(3), f(4)]), Ok(f(12)));
        assert_eq!(
            polynomial(&[f(3)]),
            Err(SumcheckError::PointLengthMismatch {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn interpolation_recovers_constant_polynomial() {
        let polynomial = interpolate_univariate(&[f(0), f(1)], &[f(5), f(5)]).unwrap();

        assert_eq!(polynomial.degree(), 0);
        assert_eq!(polynomial.evaluate(f(0)), f(5));
        assert_eq!(polynomial.evaluate(f(123)), f(5));
    }

    #[test]
    fn interpolation_recovers_linear_polynomial() {
        let polynomial = interpolate_univariate(&[f(0), f(1)], &[f(2), f(5)]).unwrap();

        assert_eq!(polynomial.coeffs(), &[f(2), f(3)]);
        assert_eq!(polynomial.evaluate(f(4)), f(14));
    }

    #[test]
    fn interpolation_recovers_quadratic_polynomial() {
        let polynomial = interpolate_univariate(&[f(0), f(1), f(2)], &[f(1), f(6), f(17)]).unwrap();

        assert_eq!(polynomial.coeffs(), &[f(1), f(2), f(3)]);
        assert_eq!(polynomial.evaluate(f(2)), f(17));
    }

    #[test]
    fn interpolation_rejects_mismatched_input_lengths() {
        assert_eq!(
            interpolate_univariate(&[f(0), f(1)], &[f(5)]),
            Err(SumcheckError::InterpolationLengthMismatch { xs: 2, ys: 1 })
        );
    }

    #[test]
    fn interpolation_rejects_empty_inputs() {
        assert_eq!(
            interpolate_univariate(&[], &[]),
            Err(SumcheckError::EmptyPolynomial)
        );
    }

    #[test]
    fn interpolation_rejects_duplicate_x_coordinates() {
        assert_eq!(
            interpolate_univariate(&[f(0), f(1), f(1)], &[f(2), f(3), f(4)]),
            Err(SumcheckError::DuplicateInterpolationPoint {
                first: 1,
                second: 2,
            })
        );
    }

    #[test]
    fn proof_container_stores_rounds_and_final_claim() {
        let round_0 = UnivariatePolynomial::new(vec![f(1), f(2)]).unwrap();
        let round_1 = UnivariatePolynomial::new(vec![f(3), f(4), f(5)]).unwrap();
        let proof = SumcheckProof::new(
            vec![round_0.clone(), round_1.clone()],
            vec![f(10), f(11)],
            f(12),
        );

        assert_eq!(proof.round_polynomials(), &[round_0, round_1]);
        assert_eq!(proof.final_point(), &[f(10), f(11)]);
        assert_eq!(proof.final_evaluation(), f(12));
        assert_eq!(proof.num_rounds(), 2);
    }
}

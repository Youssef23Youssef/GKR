//! Data structures for the Sumcheck protocol.
//!
//! Sumcheck reduces a claim about a multivariate polynomial to a sequence of
//! claims about univariate round polynomials. This module currently defines the
//! proof containers only, prover and verifier logic will be added later.

use crate::field::F;
use ark_ff::Zero;

/// Errors produced by Sumcheck data-structure validation and protocol logic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SumcheckError {
    /// A univariate polynomial must contain at least one coefficient.
    EmptyPolynomial,
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

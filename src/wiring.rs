//! Wiring predicate utilities for layered arithmetic circuits.
//!
//! This module turns a circuit layer transition into multilinear wiring
//! predicates used by GKR.
//!
//! For a transition from `previous_layer` to `current_layer`, the predicates
//! are:
//!
//! ```text
//! add_i(g, b, c)
//! mul_i(g, b, c)
//! ```
//!
//! where:
//!
//! - `g` indexes a gate in the current layer,
//! - `b` indexes the left child in the previous layer,
//! - `c` indexes the right child in the previous layer.
//!
//! Each predicate evaluates to `1` on the Boolean point corresponding to a real
//! matching gate connection, and `0` otherwise. Away from Boolean points, the
//! same functions evaluate the multilinear extension of that wiring table.

use crate::{
    circuit::{Gate, Layer},
    field::F,
    mle::{MleError, eq, index_to_bits},
};
use ark_ff::Zero;

/// Errors produced while evaluating wiring predicates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WiringError {
    /// A gate or child point has the wrong number of coordinates for its layer.
    PointLengthMismatch { expected: usize, actual: usize },

    /// An input gate appeared in a layer being treated as a computation layer.
    InputGateInComputationLayer { gate: usize },

    /// Error produced by the underlying MLE utilities.
    MleError(MleError),
}

impl From<MleError> for WiringError {
    fn from(error: MleError) -> Self {
        Self::MleError(error)
    }
}

/// Checks that `g`, `b`, and `c` have the dimensions required by the layer transition.
fn validate_point_dimensions(
    previous_layer: &Layer,
    current_layer: &Layer,
    g_point: &[F],
    b_point: &[F],
    c_point: &[F],
) -> Result<(), WiringError> {
    // `g` addresses the current layer.
    if g_point.len() != current_layer.index_bits() {
        return Err(WiringError::PointLengthMismatch {
            expected: current_layer.index_bits(),
            actual: g_point.len(),
        });
    }

    // `b` addresses the previous layer.
    if b_point.len() != previous_layer.index_bits() {
        return Err(WiringError::PointLengthMismatch {
            expected: previous_layer.index_bits(),
            actual: b_point.len(),
        });
    }

    // `c` addresses the previous layer.
    if c_point.len() != previous_layer.index_bits() {
        return Err(WiringError::PointLengthMismatch {
            expected: previous_layer.index_bits(),
            actual: c_point.len(),
        });
    }

    Ok(())
}

/// Operation selected by a wiring predicate evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateOperation {
    Add,
    Mul,
}

/// Evaluates either the Add or Mul wiring predicate for a layer transition.
fn evaluate_gate_predicate(
    previous_layer: &Layer,
    current_layer: &Layer,
    g_point: &[F],
    b_point: &[F],
    c_point: &[F],
    operation: GateOperation,
) -> Result<F, WiringError> {
    validate_point_dimensions(previous_layer, current_layer, g_point, b_point, c_point)?;

    let mut result = F::zero();

    // Sum contributions from real gates matching the selected operation.
    for (gate_index, gate) in current_layer.gates().iter().enumerate() {
        let (left, right) = match (operation, gate) {
            (GateOperation::Add, Gate::Add { left, right }) => (*left, *right),
            (GateOperation::Mul, Gate::Mul { left, right }) => (*left, *right),

            (_, Gate::Add { .. }) | (_, Gate::Mul { .. }) => continue,

            (_, Gate::Input { .. }) => {
                return Err(WiringError::InputGateInComputationLayer { gate: gate_index });
            }
        };

        let gate_bits = index_to_bits(gate_index, current_layer.index_bits())?;
        let left_bits = index_to_bits(left, previous_layer.index_bits())?;
        let right_bits = index_to_bits(right, previous_layer.index_bits())?;

        let contribution =
            eq(g_point, &gate_bits)? * eq(b_point, &left_bits)? * eq(c_point, &right_bits)?;

        result += contribution;
    }

    Ok(result)
}

/// Public wrapper for Add wiring predicate `add_i(g, b, c)`.
pub fn evaluate_add_predicate(
    previous_layer: &Layer,
    current_layer: &Layer,
    g_point: &[F],
    b_point: &[F],
    c_point: &[F],
) -> Result<F, WiringError> {
    evaluate_gate_predicate(
        previous_layer,
        current_layer,
        g_point,
        b_point,
        c_point,
        GateOperation::Add,
    )
}

/// Public wrapper Mul wiring predicate `mul_i(g, b, c)`.
pub fn evaluate_mul_predicate(
    previous_layer: &Layer,
    current_layer: &Layer,
    g_point: &[F],
    b_point: &[F],
    c_point: &[F],
) -> Result<F, WiringError> {
    evaluate_gate_predicate(
        previous_layer,
        current_layer,
        g_point,
        b_point,
        c_point,
        GateOperation::Mul,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::Zero;

    fn f(value: u64) -> F {
        F::from(value)
    }

    fn layer(gates: Vec<Gate>) -> Layer {
        Layer::new(gates).unwrap()
    }

    fn target_layers() -> (Layer, Layer, Layer) {
        let layer0 = layer(vec![
            Gate::Input { input_index: 0 },
            Gate::Input { input_index: 1 },
            Gate::Input { input_index: 2 },
            Gate::Input { input_index: 3 },
        ]);

        let layer1 = layer(vec![
            Gate::Add { left: 0, right: 1 },
            Gate::Add { left: 2, right: 3 },
        ]);

        let layer2 = layer(vec![Gate::Mul { left: 0, right: 1 }]);

        (layer0, layer1, layer2)
    }

    #[test]
    fn add_predicate_matches_boolean_wiring_points() {
        let (layer0, layer1, _) = target_layers();

        assert_eq!(
            evaluate_add_predicate(&layer0, &layer1, &[f(0)], &[f(0), f(0)], &[f(1), f(0)]),
            Ok(f(1))
        );

        assert_eq!(
            evaluate_add_predicate(&layer0, &layer1, &[f(1)], &[f(0), f(1)], &[f(1), f(1)]),
            Ok(f(1))
        );

        assert_eq!(
            evaluate_add_predicate(&layer0, &layer1, &[f(0)], &[f(1), f(0)], &[f(0), f(0)]),
            Ok(F::zero())
        );
    }

    #[test]
    fn mul_predicate_is_zero_on_add_layer() {
        let (layer0, layer1, _) = target_layers();

        assert_eq!(
            evaluate_mul_predicate(&layer0, &layer1, &[f(0)], &[f(0), f(0)], &[f(1), f(0)]),
            Ok(F::zero())
        );
    }

    #[test]
    fn predicates_match_second_layer_boolean_points() {
        let (_, layer1, layer2) = target_layers();

        assert_eq!(
            evaluate_mul_predicate(&layer1, &layer2, &[], &[f(0)], &[f(1)]),
            Ok(f(1))
        );

        assert_eq!(
            evaluate_add_predicate(&layer1, &layer2, &[], &[f(0)], &[f(1)]),
            Ok(F::zero())
        );
    }

    #[test]
    fn predicates_are_zero_at_padded_gate_positions() {
        let previous_layer = layer(vec![
            Gate::Input { input_index: 0 },
            Gate::Input { input_index: 1 },
        ]);

        let current_layer = layer(vec![
            Gate::Add { left: 0, right: 1 },
            Gate::Mul { left: 0, right: 1 },
            Gate::Add { left: 1, right: 0 },
        ]);

        // Current layer has real width 3 and padded width 4.
        // Padded gate index 3 is addressed by little-endian bits [1, 1].
        let padded_gate_point = [f(1), f(1)];

        assert_eq!(
            evaluate_add_predicate(
                &previous_layer,
                &current_layer,
                &padded_gate_point,
                &[f(0)],
                &[f(1)],
            ),
            Ok(F::zero())
        );

        assert_eq!(
            evaluate_mul_predicate(
                &previous_layer,
                &current_layer,
                &padded_gate_point,
                &[f(0)],
                &[f(1)],
            ),
            Ok(F::zero())
        );
    }

    #[test]
    fn add_predicate_interpolates_at_non_boolean_gate_point() {
        let previous_layer = layer(vec![
            Gate::Input { input_index: 0 },
            Gate::Input { input_index: 1 },
        ]);

        let current_layer = layer(vec![
            Gate::Add { left: 0, right: 1 },
            Gate::Add { left: 1, right: 0 },
        ]);

        let r = f(3);

        assert_eq!(
            evaluate_add_predicate(&previous_layer, &current_layer, &[r], &[f(0)], &[f(1)]),
            Ok(f(1) - r)
        );

        assert_eq!(
            evaluate_add_predicate(&previous_layer, &current_layer, &[r], &[f(1)], &[f(0)]),
            Ok(r)
        );
    }
}

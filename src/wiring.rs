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
    mle::{MleError, eq, evaluate_mle, index_to_bits},
};
use ark_ff::Zero;

/// Errors produced while evaluating wiring predicates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WiringError {
    /// A gate or child point has the wrong number of coordinates for its layer.
    PointLengthMismatch { expected: usize, actual: usize },

    /// An input gate appeared in a layer being treated as a computation layer.
    InputGateInComputationLayer { gate: usize },

    /// Previous-layer values do not match the previous layer's padded width.
    PreviousValuesLengthMismatch { expected: usize, actual: usize },

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

/// Evaluates the GKR layer equation at `(g, b, c)`.
///
/// For a transition from `previous_layer` to `current_layer`, this computes:
///
/// ```text
/// add_i(g,b,c) * (V_{i-1}(b) + V_{i-1}(c))
///   + mul_i(g,b,c) * V_{i-1}(b) * V_{i-1}(c)
/// ```
///
/// where `V_{i-1}` is the multilinear extension represented by
/// `previous_values`.
pub fn evaluate_layer_equation(
    previous_layer: &Layer,
    current_layer: &Layer,
    previous_values: &[F],
    g_point: &[F],
    b_point: &[F],
    c_point: &[F],
) -> Result<F, WiringError> {
    if previous_values.len() != previous_layer.padded_width() {
        return Err(WiringError::PreviousValuesLengthMismatch {
            expected: previous_layer.padded_width(),
            actual: previous_values.len(),
        });
    }

    let add = evaluate_gate_predicate(
        previous_layer,
        current_layer,
        g_point,
        b_point,
        c_point,
        GateOperation::Add,
    )?;
    let mul = evaluate_gate_predicate(
        previous_layer,
        current_layer,
        g_point,
        b_point,
        c_point,
        GateOperation::Mul,
    )?;

    let left_value = evaluate_mle(previous_values, b_point)?;
    let right_value = evaluate_mle(previous_values, c_point)?;

    Ok(add * (left_value + right_value) + mul * left_value * right_value)
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
            evaluate_gate_predicate(
                &layer0,
                &layer1,
                &[f(0)],
                &[f(0), f(0)],
                &[f(1), f(0)],
                GateOperation::Add,
            ),
            Ok(f(1))
        );

        assert_eq!(
            evaluate_gate_predicate(
                &layer0,
                &layer1,
                &[f(1)],
                &[f(0), f(1)],
                &[f(1), f(1)],
                GateOperation::Add,
            ),
            Ok(f(1))
        );

        assert_eq!(
            evaluate_gate_predicate(
                &layer0,
                &layer1,
                &[f(0)],
                &[f(1), f(0)],
                &[f(0), f(0)],
                GateOperation::Add,
            ),
            Ok(F::zero())
        );
    }

    #[test]
    fn mul_predicate_is_zero_on_add_layer() {
        let (layer0, layer1, _) = target_layers();

        assert_eq!(
            evaluate_gate_predicate(
                &layer0,
                &layer1,
                &[f(0)],
                &[f(0), f(0)],
                &[f(1), f(0)],
                GateOperation::Mul,
            ),
            Ok(F::zero())
        );
    }

    #[test]
    fn predicates_match_second_layer_boolean_points() {
        let (_, layer1, layer2) = target_layers();

        assert_eq!(
            evaluate_gate_predicate(&layer1, &layer2, &[], &[f(0)], &[f(1)], GateOperation::Mul,),
            Ok(f(1))
        );

        assert_eq!(
            evaluate_gate_predicate(&layer1, &layer2, &[], &[f(0)], &[f(1)], GateOperation::Add,),
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
            evaluate_gate_predicate(
                &previous_layer,
                &current_layer,
                &padded_gate_point,
                &[f(0)],
                &[f(1)],
                GateOperation::Add,
            ),
            Ok(F::zero())
        );

        assert_eq!(
            evaluate_gate_predicate(
                &previous_layer,
                &current_layer,
                &padded_gate_point,
                &[f(0)],
                &[f(1)],
                GateOperation::Mul,
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
            evaluate_gate_predicate(
                &previous_layer,
                &current_layer,
                &[r],
                &[f(0)],
                &[f(1)],
                GateOperation::Add,
            ),
            Ok(f(1) - r)
        );

        assert_eq!(
            evaluate_gate_predicate(
                &previous_layer,
                &current_layer,
                &[r],
                &[f(1)],
                &[f(0)],
                GateOperation::Add,
            ),
            Ok(r)
        );
    }

    #[test]
    fn layer_equation_evaluates_add_gate_at_boolean_points() {
        let (layer0, layer1, _) = target_layers();
        let previous_values = vec![f(2), f(3), f(5), f(7)];

        let value = evaluate_layer_equation(
            &layer0,
            &layer1,
            &previous_values,
            &[f(0)],
            &[f(0), f(0)],
            &[f(1), f(0)],
        );

        assert_eq!(value, Ok(f(5)));
    }

    #[test]
    fn layer_equation_evaluates_mul_gate_at_boolean_points() {
        let (_, layer1, layer2) = target_layers();
        let previous_values = vec![f(5), f(12)];

        let value =
            evaluate_layer_equation(&layer1, &layer2, &previous_values, &[], &[f(0)], &[f(1)]);

        assert_eq!(value, Ok(f(60)));
    }

    #[test]
    fn layer_equation_returns_zero_for_wrong_wiring() {
        let (layer0, layer1, _) = target_layers();
        let previous_values = vec![f(2), f(3), f(5), f(7)];

        let value = evaluate_layer_equation(
            &layer0,
            &layer1,
            &previous_values,
            &[f(0)],
            &[f(1), f(0)],
            &[f(0), f(0)],
        );

        assert_eq!(value, Ok(F::zero()));
    }

    #[test]
    fn layer_equation_rejects_wrong_previous_values_length() {
        let (layer0, layer1, _) = target_layers();
        let previous_values = vec![f(2), f(3), f(5)];

        let value = evaluate_layer_equation(
            &layer0,
            &layer1,
            &previous_values,
            &[f(0)],
            &[f(0), f(0)],
            &[f(1), f(0)],
        );

        assert_eq!(
            value,
            Err(WiringError::PreviousValuesLengthMismatch {
                expected: 4,
                actual: 3,
            })
        );
    }
}

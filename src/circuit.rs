use crate::field::F;

/// A gate in a layered arithmetic circuit.
///
/// A `Gate` describes wiring, not a runtime value. Actual field values are
/// stored later in `CircuitEvaluation`.
///
/// Input gates:
/// - `Input { input_index }` is only valid in layer 0.
/// - `input_index` means "which external input value this gate reads".
/// - It does not mean "how many inputs exist".
///
/// For example, `Input { input_index: 2 }` means:
///
/// ```text
/// this gate reads inputs[2]
/// ```
///
/// Computation gates:
/// - `Add` and `Mul` gates are valid only after layer 0.
/// - Their `left` and `right` fields are indices into the immediately
///   preceding layer.
/// - They do not store child layer numbers.
///
/// For example, if an `Add` or `Mul` gate is in layer `i`, then:
///
/// ```text
/// left  refers to layers[i - 1][left]
/// right refers to layers[i - 1][right]
/// ```
///
/// This intentionally prevents gates from directly referencing non-adjacent
/// layers, which keeps the circuit compatible with the layer-by-layer structure
/// used by GKR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gate {
    Input { input_index: usize },
    Add { left: usize, right: usize },
    Mul { left: usize, right: usize },
}

/// A single layer of gates in a layered arithmetic circuit.
///
/// A layer contains only the real gates of the circuit. Padding gates are not
/// stored here. During evaluation, the value vector for this layer is extended
/// to `padded_width`, and all padded positions are filled with zero.
///
/// These width values are needed later when constructing multilinear extensions for GKR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layer {
    /// Real gates only, padded positions are not stored as fake gates.
    gates: Vec<Gate>,

    /// Number of real gates, equal to `gates.len()`.
    real_width: usize,

    /// Width rounded up to the next power of two.
    padded_width: usize,

    /// Number of Boolean variables needed to index the padded layer.
    index_bits: usize,
}

/// A complete layered arithmetic circuit.
///
/// The circuit is stored as an ordered list of layers:
///
/// - layer 0 is the input layer
/// - layer 1 is the first computation layer
/// - the final layer is the output layer
///
/// Evaluation runs forward from the input layer to the output layer. Later, GKR
/// verification will work in the opposite direction, reducing claims from the
/// output layer back toward the input layer.
///
/// - `layers` contains all circuit layers in input-to-output order.
/// - `expected_inputs` is the exact number of public input values expected by
///   the input layer.
///
/// For the basic GKR version, the final layer should contain exactly one real
/// output gate.
pub struct Circuit {
    layers: Vec<Layer>,
    expected_inputs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitError {
    EmptyLayer,
}

/// The result of evaluating a circuit over the chosen finite field.
///
/// `Circuit` stores the wiring/blueprint of the computation, while
/// `CircuitEvaluation` stores the actual field values produced by running that
/// circuit on concrete inputs.
///
/// - `layer_values[i]` contains the evaluated values for layer `i`.
/// - Each inner vector is padded to that layer's `padded_width`.
/// - Real gate outputs occupy the first `real_width` positions.
/// - Padded positions are filled with zero.
/// - `output` is the value at index 0 of the final layer.
///
/// These evaluated layer values will later become the GKR prover's witness.
pub struct CircuitEvaluation {
    layer_values: Vec<Vec<F>>,
    output: F,
}


impl Layer {
    pub fn new(gates: Vec<Gate>) -> Result<Self, CircuitError> {
        if gates.is_empty() {
            return Err(CircuitError::EmptyLayer);
        }

        // Number of real, non-padding gates in this layer.
        let real_width = gates.len();

        // Round the real width up to the next power of two.
        //
        // GKR/MLE code later works over Boolean hypercubes of size 2^n, so each layer
        // gets a power-of-two evaluation domain. Since empty layers are rejected above,
        // `real_width` is always at least 1 here.
        let padded_width = real_width.next_power_of_two();

        // Number of bits needed to index the padded layer.
        //
        // Because `padded_width` is a power of two, this is log2(padded_width).
        // Examples:
        // - padded_width = 1 => index_bits = 0
        // - padded_width = 2 => index_bits = 1
        // - padded_width = 4 => index_bits = 2
        let index_bits = padded_width.trailing_zeros() as usize;

        Ok(Self {
            gates,
            real_width,
            padded_width,
            index_bits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_with_one_gate_has_width_one_and_zero_index_bits() {
        let layer = Layer::new(vec![
            Gate::Input { input_index: 0 },
        ]).unwrap();

        assert_eq!(layer.real_width, 1);
        assert_eq!(layer.padded_width, 1);
        assert_eq!(layer.index_bits, 0);
    }

    #[test]
    fn layer_with_two_gates_has_width_two_and_one_index_bit() {
        let layer = Layer::new(vec![
            Gate::Input { input_index: 0 },
            Gate::Input { input_index: 1 },
        ]).unwrap();

        assert_eq!(layer.real_width, 2);
        assert_eq!(layer.padded_width, 2);
        assert_eq!(layer.index_bits, 1);
    }

    #[test]
    fn layer_with_three_gates_is_padded_to_four() {
        let layer = Layer::new(vec![
            Gate::Input { input_index: 0 },
            Gate::Input { input_index: 1 },
            Gate::Input { input_index: 2 },
        ]).unwrap();

        assert_eq!(layer.real_width, 3);
        assert_eq!(layer.padded_width, 4);
        assert_eq!(layer.index_bits, 2);
    }

    #[test]
    fn layer_with_five_gates_is_padded_to_eight() {
        let layer = Layer::new(vec![
            Gate::Input { input_index: 0 },
            Gate::Input { input_index: 1 },
            Gate::Input { input_index: 2 },
            Gate::Input { input_index: 3 },
            Gate::Input { input_index: 4 },
        ]).unwrap();

        assert_eq!(layer.real_width, 5);
        assert_eq!(layer.padded_width, 8);
        assert_eq!(layer.index_bits, 3);
    }

    #[test]
    fn empty_layer_is_rejected() {
        let result = Layer::new(vec![]);

        assert_eq!(result, Err(CircuitError::EmptyLayer));
    }
}

// TODO Phase 3: Implement validation
//! Layered arithmetic circuit representation and evaluation.
//!
//! Circuits are stored from inputs to outputs: layer `0` is the public input
//! layer, and each later layer contains gates whose children live in the
//! immediately preceding layer. Evaluation runs in that same direction and
//! stores every padded layer value vector as the prover witness.
//!
//! Padding is implicit: layers store only real gates, while evaluation vectors
//! are extended to the next power of two with zero values. This keeps the
//! circuit model simple while producing tables that can be used directly as
//! multilinear-extension evaluations.

use crate::field::F;
use ark_ff::Zero;

/// A real gate in a layered arithmetic circuit.
///
/// Input gates are valid only in layer `0`. Addition and multiplication gates
/// reference child indices in the immediately preceding layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gate {
    /// Reads one public input by index.
    Input { input_index: usize },

    /// Adds two values from the previous layer.
    Add { left: usize, right: usize },

    /// Multiplies two values from the previous layer.
    Mul { left: usize, right: usize },
}

/// Errors produced while constructing, validating, or evaluating circuits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitError {
    /// A valid circuit needs at least one input layer and one output layer.
    NotEnoughLayers,

    /// Layers must contain at least one real gate.
    EmptyLayer,

    /// Stored layer metadata does not match the number of real gates.
    InvalidRealWidth { layer: usize },

    /// Padded width is not the next power of two for the real width.
    InvalidPaddedWidth { layer: usize },

    /// Index-bit metadata does not match `log2(padded_width)`.
    InvalidIndexBits { layer: usize },

    /// Layer `0` must contain only input gates.
    InvalidInputLayer,

    /// An input gate references an input outside the expected input vector.
    InvalidInputIndex { gate: usize, input_index: usize },

    /// A computation layer contains a gate kind that is not allowed there.
    InvalidComputationGate { layer: usize, gate: usize },

    /// A child index does not refer to a real gate in the previous layer.
    InvalidChildIndex {
        layer: usize,
        gate: usize,
        child: usize,
    },

    /// The provided input count does not match the circuit input layer.
    WrongInputCount { expected: usize, actual: usize },

    /// Input gates must appear in the same order as the public input vector.
    InvalidInputGateOrder { gate: usize, input_index: usize },
}

/// Field values produced by evaluating a circuit on concrete inputs.
///
/// `layer_values` stores every padded layer table. `outputs` contains the real
/// values from the final layer. These values form the circuit witness used by
/// later GKR phases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CircuitEvaluation {
    layer_values: Vec<Vec<F>>,
    outputs: Vec<F>,
}

impl CircuitEvaluation {
    /// Returns the real output values from the final layer.
    pub fn outputs(&self) -> &[F] {
        &self.outputs
    }

    /// Returns all padded layer evaluation tables.
    pub fn layer_values(&self) -> &[Vec<F>] {
        &self.layer_values
    }
}

/// A layer of real circuit gates plus derived padding metadata.
///
/// Padding gates are not stored. During evaluation, each layer receives a value
/// vector of length `padded_width`, with unused positions set to zero.
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

impl Layer {
    /// Creates a layer and derives its width metadata.
    pub fn new(gates: Vec<Gate>) -> Result<Self, CircuitError> {
        if gates.is_empty() {
            return Err(CircuitError::EmptyLayer);
        }

        let real_width = gates.len();
        let padded_width = real_width.next_power_of_two();
        let index_bits = padded_width.trailing_zeros() as usize;

        Ok(Self {
            gates,
            real_width,
            padded_width,
            index_bits,
        })
    }

    /// Returns the real gates stored in this layer.
    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }

    /// Returns the number of real gates.
    pub fn real_width(&self) -> usize {
        self.real_width
    }

    /// Returns the power-of-two width used by MLE tables.
    pub fn padded_width(&self) -> usize {
        self.padded_width
    }

    /// Returns the number of Boolean variables needed to index the padded layer.
    pub fn index_bits(&self) -> usize {
        self.index_bits
    }

    /// Validates that stored metadata is consistent with the real gates.
    fn validate_metadata(&self, layer_index: usize) -> Result<(), CircuitError> {
        if self.real_width == 0 {
            return Err(CircuitError::EmptyLayer);
        }

        if self.real_width != self.gates.len() {
            return Err(CircuitError::InvalidRealWidth { layer: layer_index });
        }

        if self.padded_width == 0 || !self.padded_width.is_power_of_two() {
            return Err(CircuitError::InvalidPaddedWidth { layer: layer_index });
        }

        if self.padded_width != self.real_width.next_power_of_two() {
            return Err(CircuitError::InvalidPaddedWidth { layer: layer_index });
        }

        let expected_index_bits = self.padded_width.ilog2() as usize;

        if self.index_bits != expected_index_bits {
            return Err(CircuitError::InvalidIndexBits { layer: layer_index });
        }

        Ok(())
    }
}

/// A complete layered arithmetic circuit.
///
/// Layers are stored in input-to-output order. Computation gates may reference
/// only real gates in the immediately preceding layer. The final layer may
/// contain one or more real output gates.
pub struct Circuit {
    layers: Vec<Layer>,
    expected_inputs: usize,
}

impl Circuit {
    /// Creates and validates a circuit.
    pub fn new(layers: Vec<Layer>, expected_inputs: usize) -> Result<Self, CircuitError> {
        let circuit = Self {
            layers,
            expected_inputs,
        };

        circuit.validate()?;

        Ok(circuit)
    }

    /// Validates circuit shape, layer metadata, input gates, and child wiring.
    pub fn validate(&self) -> Result<(), CircuitError> {
        // Require both an input layer and an output layer.
        if self.layers.len() < 2 {
            return Err(CircuitError::NotEnoughLayers);
        }

        // Validate derived metadata before using layer widths for later checks.
        for (layer_index, layer) in self.layers.iter().enumerate() {
            layer.validate_metadata(layer_index)?;
        }

        // The input layer must match the public input vector exactly.
        if self.layers[0].real_width != self.expected_inputs {
            return Err(CircuitError::WrongInputCount {
                expected: self.expected_inputs,
                actual: self.layers[0].real_width,
            });
        }

        for (gate_index, gate) in self.layers[0].gates.iter().enumerate() {
            match gate {
                Gate::Input { input_index } => {
                    if *input_index >= self.expected_inputs {
                        return Err(CircuitError::InvalidInputIndex {
                            gate: gate_index,
                            input_index: *input_index,
                        });
                    }

                    if *input_index != gate_index {
                        return Err(CircuitError::InvalidInputGateOrder {
                            gate: gate_index,
                            input_index: *input_index,
                        });
                    }
                }

                _ => {
                    return Err(CircuitError::InvalidInputLayer);
                }
            }
        }

        // Computation gates may reference only real gates in the previous layer.
        for layer_index in 1..self.layers.len() {
            let previous_layer = &self.layers[layer_index - 1];
            let current_layer = &self.layers[layer_index];

            for (gate_index, gate) in current_layer.gates.iter().enumerate() {
                match gate {
                    Gate::Add { left, right } | Gate::Mul { left, right } => {
                        if *left >= previous_layer.real_width {
                            return Err(CircuitError::InvalidChildIndex {
                                layer: layer_index,
                                gate: gate_index,
                                child: *left,
                            });
                        }

                        if *right >= previous_layer.real_width {
                            return Err(CircuitError::InvalidChildIndex {
                                layer: layer_index,
                                gate: gate_index,
                                child: *right,
                            });
                        }
                    }

                    Gate::Input { .. } => {
                        return Err(CircuitError::InvalidComputationGate {
                            layer: layer_index,
                            gate: gate_index,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Evaluates the circuit over the field and returns all padded layer values.
    pub fn evaluate(&self, inputs: &[F]) -> Result<CircuitEvaluation, CircuitError> {
        self.validate()?;

        if inputs.len() != self.expected_inputs {
            return Err(CircuitError::WrongInputCount {
                expected: self.expected_inputs,
                actual: inputs.len(),
            });
        }

        // Input values occupy real input positions; padding remains zero.
        let input_layer = &self.layers[0];
        let mut input_values = vec![F::zero(); input_layer.padded_width];

        for (gate_index, gate) in input_layer.gates.iter().enumerate() {
            match gate {
                Gate::Input { input_index } => {
                    input_values[gate_index] = inputs[*input_index];
                }

                _ => unreachable!("validate() ensures layer 0 contains only input gates"),
            }
        }

        let mut layer_values = Vec::new();
        layer_values.push(input_values);

        // Evaluate computation layers and keep each padded table as witness data.
        for layer_index in 1..self.layers.len() {
            let current_layer = &self.layers[layer_index];
            let previous_values = layer_values
                .last()
                .expect("there is always a previous evaluated layer");

            let mut current_values = vec![F::zero(); current_layer.padded_width];

            for (gate_index, gate) in current_layer.gates.iter().enumerate() {
                match gate {
                    Gate::Add { left, right } => {
                        current_values[gate_index] =
                            previous_values[*left] + previous_values[*right];
                    }

                    Gate::Mul { left, right } => {
                        current_values[gate_index] =
                            previous_values[*left] * previous_values[*right];
                    }

                    Gate::Input { .. } => {
                        unreachable!("validate() ensures computation layers contain no input gates")
                    }
                }
            }

            layer_values.push(current_values);
        }

        let final_layer = self
            .layers
            .last()
            .expect("validate() ensures the circuit has an output layer");

        let final_values = layer_values
            .last()
            .expect("validate() ensures the circuit has an output layer");

        let outputs = final_values[..final_layer.real_width].to_vec();

        Ok(CircuitEvaluation {
            layer_values,
            outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_with_one_gate_has_width_one_and_zero_index_bits() {
        let layer = Layer::new(vec![Gate::Input { input_index: 0 }]).unwrap();

        assert_eq!(layer.real_width, 1);
        assert_eq!(layer.padded_width, 1);
        assert_eq!(layer.index_bits, 0);
    }

    #[test]
    fn layer_with_two_gates_has_width_two_and_one_index_bit() {
        let layer = Layer::new(vec![
            Gate::Input { input_index: 0 },
            Gate::Input { input_index: 1 },
        ])
        .unwrap();

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
        ])
        .unwrap();

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
        ])
        .unwrap();

        assert_eq!(layer.real_width, 5);
        assert_eq!(layer.padded_width, 8);
        assert_eq!(layer.index_bits, 3);
    }

    #[test]
    fn empty_layer_is_rejected() {
        let result = Layer::new(vec![]);

        assert_eq!(result, Err(CircuitError::EmptyLayer));
    }

    // Circuit validation tests

    fn layer(gates: Vec<Gate>) -> Layer {
        Layer::new(gates).unwrap()
    }

    fn target_circuit() -> Circuit {
        Circuit {
            layers: vec![
                layer(vec![
                    Gate::Input { input_index: 0 },
                    Gate::Input { input_index: 1 },
                    Gate::Input { input_index: 2 },
                    Gate::Input { input_index: 3 },
                ]),
                layer(vec![
                    Gate::Add { left: 0, right: 1 },
                    Gate::Add { left: 2, right: 3 },
                ]),
                layer(vec![Gate::Mul { left: 0, right: 1 }]),
            ],
            expected_inputs: 4,
        }
    }

    #[test]
    fn valid_target_circuit_passes_validation() {
        let circuit = target_circuit();

        assert_eq!(circuit.validate(), Ok(()));
    }

    #[test]
    fn target_circuit_layer_widths_are_correct() {
        let circuit = target_circuit();

        assert_eq!(circuit.layers[0].real_width, 4);
        assert_eq!(circuit.layers[0].padded_width, 4);
        assert_eq!(circuit.layers[0].index_bits, 2);

        assert_eq!(circuit.layers[1].real_width, 2);
        assert_eq!(circuit.layers[1].padded_width, 2);
        assert_eq!(circuit.layers[1].index_bits, 1);

        assert_eq!(circuit.layers[2].real_width, 1);
        assert_eq!(circuit.layers[2].padded_width, 1);
        assert_eq!(circuit.layers[2].index_bits, 0);
    }

    #[test]
    fn empty_computation_layer_is_rejected_by_validation() {
        let circuit = Circuit {
            layers: vec![
                layer(vec![Gate::Input { input_index: 0 }]),
                Layer {
                    gates: vec![],
                    real_width: 0,
                    padded_width: 1,
                    index_bits: 0,
                },
            ],
            expected_inputs: 1,
        };

        assert_eq!(circuit.validate(), Err(CircuitError::EmptyLayer));
    }

    #[test]
    fn invalid_real_width_is_rejected_by_validation() {
        let circuit = Circuit {
            layers: vec![
                Layer {
                    gates: vec![Gate::Input { input_index: 0 }],
                    real_width: 2,
                    padded_width: 2,
                    index_bits: 1,
                },
                layer(vec![Gate::Add { left: 0, right: 0 }]),
            ],
            expected_inputs: 1,
        };

        assert_eq!(
            circuit.validate(),
            Err(CircuitError::InvalidRealWidth { layer: 0 })
        );
    }

    #[test]
    fn invalid_padded_width_is_rejected_by_validation() {
        let circuit = Circuit {
            layers: vec![
                Layer {
                    gates: vec![
                        Gate::Input { input_index: 0 },
                        Gate::Input { input_index: 1 },
                    ],
                    real_width: 2,
                    padded_width: 3,
                    index_bits: 1,
                },
                layer(vec![Gate::Add { left: 0, right: 1 }]),
            ],
            expected_inputs: 2,
        };

        assert_eq!(
            circuit.validate(),
            Err(CircuitError::InvalidPaddedWidth { layer: 0 })
        );
    }

    #[test]
    fn invalid_index_bits_are_rejected_by_validation() {
        let circuit = Circuit {
            layers: vec![
                Layer {
                    gates: vec![
                        Gate::Input { input_index: 0 },
                        Gate::Input { input_index: 1 },
                    ],
                    real_width: 2,
                    padded_width: 2,
                    index_bits: 2,
                },
                layer(vec![Gate::Add { left: 0, right: 1 }]),
            ],
            expected_inputs: 2,
        };

        assert_eq!(
            circuit.validate(),
            Err(CircuitError::InvalidIndexBits { layer: 0 })
        );
    }

    #[test]
    fn circuit_with_only_one_layer_is_rejected() {
        let circuit = Circuit {
            layers: vec![layer(vec![Gate::Input { input_index: 0 }])],
            expected_inputs: 1,
        };

        assert_eq!(circuit.validate(), Err(CircuitError::NotEnoughLayers));
    }

    #[test]
    fn input_layer_must_contain_only_input_gates() {
        let circuit = Circuit {
            layers: vec![
                layer(vec![Gate::Add { left: 0, right: 0 }]),
                layer(vec![Gate::Mul { left: 0, right: 0 }]),
            ],
            expected_inputs: 1,
        };

        assert_eq!(circuit.validate(), Err(CircuitError::InvalidInputLayer));
    }

    #[test]
    fn input_layer_width_must_match_expected_inputs() {
        let circuit = Circuit {
            layers: vec![
                layer(vec![
                    Gate::Input { input_index: 0 },
                    Gate::Input { input_index: 1 },
                ]),
                layer(vec![Gate::Add { left: 0, right: 1 }]),
            ],
            expected_inputs: 3,
        };

        assert_eq!(
            circuit.validate(),
            Err(CircuitError::WrongInputCount {
                expected: 3,
                actual: 2,
            })
        );
    }

    #[test]
    fn input_gate_must_reference_existing_input() {
        let circuit = Circuit {
            layers: vec![
                layer(vec![
                    Gate::Input { input_index: 0 },
                    Gate::Input { input_index: 2 },
                ]),
                layer(vec![Gate::Add { left: 0, right: 1 }]),
            ],
            expected_inputs: 2,
        };

        assert_eq!(
            circuit.validate(),
            Err(CircuitError::InvalidInputIndex {
                gate: 1,
                input_index: 2,
            })
        );
    }

    #[test]
    fn computation_layers_must_not_contain_input_gates() {
        let circuit = Circuit {
            layers: vec![
                layer(vec![Gate::Input { input_index: 0 }]),
                layer(vec![Gate::Input { input_index: 0 }]),
            ],
            expected_inputs: 1,
        };

        assert_eq!(
            circuit.validate(),
            Err(CircuitError::InvalidComputationGate { layer: 1, gate: 0 })
        );
    }

    #[test]
    fn child_indices_must_reference_previous_real_width() {
        let circuit = Circuit {
            layers: vec![
                layer(vec![
                    Gate::Input { input_index: 0 },
                    Gate::Input { input_index: 1 },
                    Gate::Input { input_index: 2 },
                ]),
                layer(vec![Gate::Add { left: 0, right: 3 }]),
            ],
            expected_inputs: 3,
        };

        assert_eq!(
            circuit.validate(),
            Err(CircuitError::InvalidChildIndex {
                layer: 1,
                gate: 0,
                child: 3,
            })
        );
    }

    #[test]
    fn final_layer_may_have_multiple_outputs() {
        let circuit = Circuit::new(
            vec![
                layer(vec![
                    Gate::Input { input_index: 0 },
                    Gate::Input { input_index: 1 },
                    Gate::Input { input_index: 2 },
                    Gate::Input { input_index: 3 },
                ]),
                layer(vec![
                    Gate::Add { left: 0, right: 1 },
                    Gate::Mul { left: 2, right: 3 },
                ]),
            ],
            4,
        )
        .unwrap();

        let inputs = vec![F::from(2u64), F::from(3u64), F::from(5u64), F::from(7u64)];

        let evaluation = circuit.evaluate(&inputs).unwrap();

        assert_eq!(evaluation.outputs(), &[F::from(5u64), F::from(35u64)]);
    }

    // Circuit evaluation tests

    #[test]
    fn target_circuit_evaluates_expected_output() {
        let circuit = target_circuit();
        let inputs = vec![F::from(2u64), F::from(3u64), F::from(5u64), F::from(7u64)];

        let evaluation = circuit.evaluate(&inputs).unwrap();

        assert_eq!(evaluation.outputs(), &[F::from(60u64)]);
    }

    #[test]
    fn target_circuit_stores_all_layer_values() {
        let circuit = target_circuit();
        let inputs = vec![F::from(2u64), F::from(3u64), F::from(5u64), F::from(7u64)];

        let evaluation = circuit.evaluate(&inputs).unwrap();
        let layer_values = evaluation.layer_values();

        assert_eq!(layer_values.len(), 3);
        assert_eq!(
            layer_values[0],
            vec![F::from(2u64), F::from(3u64), F::from(5u64), F::from(7u64),]
        );
        assert_eq!(layer_values[1], vec![F::from(5u64), F::from(12u64)]);
        assert_eq!(layer_values[2], vec![F::from(60u64)]);
    }

    #[test]
    fn evaluation_keeps_padded_positions_zero() {
        let circuit = Circuit::new(
            vec![
                layer(vec![
                    Gate::Input { input_index: 0 },
                    Gate::Input { input_index: 1 },
                    Gate::Input { input_index: 2 },
                    Gate::Input { input_index: 3 },
                    Gate::Input { input_index: 4 },
                    Gate::Input { input_index: 5 },
                ]),
                layer(vec![
                    Gate::Add { left: 0, right: 1 },
                    Gate::Add { left: 2, right: 3 },
                    Gate::Add { left: 4, right: 5 },
                ]),
                layer(vec![Gate::Mul { left: 0, right: 2 }]),
            ],
            6,
        )
        .unwrap();

        let inputs = vec![
            F::from(1u64),
            F::from(2u64),
            F::from(3u64),
            F::from(4u64),
            F::from(5u64),
            F::from(6u64),
        ];

        let evaluation = circuit.evaluate(&inputs).unwrap();
        let layer_values = evaluation.layer_values();

        assert_eq!(layer_values[1].len(), 4);
        assert_eq!(layer_values[1][0], F::from(3u64));
        assert_eq!(layer_values[1][1], F::from(7u64));
        assert_eq!(layer_values[1][2], F::from(11u64));
        assert_eq!(layer_values[1][3], F::zero());
        assert_eq!(evaluation.outputs(), &[F::from(33u64)]);
    }

    #[test]
    fn evaluation_rejects_wrong_input_count() {
        let circuit = target_circuit();
        let inputs = vec![F::from(2u64), F::from(3u64), F::from(5u64)];

        assert_eq!(
            circuit.evaluate(&inputs),
            Err(CircuitError::WrongInputCount {
                expected: 4,
                actual: 3,
            })
        );
    }

    #[test]
    fn evaluation_rejects_invalid_circuit() {
        let circuit = Circuit {
            layers: vec![
                layer(vec![
                    Gate::Input { input_index: 0 },
                    Gate::Input { input_index: 1 },
                ]),
                layer(vec![Gate::Add { left: 0, right: 2 }]),
            ],
            expected_inputs: 2,
        };

        let inputs = vec![F::from(2u64), F::from(3u64)];

        assert_eq!(
            circuit.evaluate(&inputs),
            Err(CircuitError::InvalidChildIndex {
                layer: 1,
                gate: 0,
                child: 2,
            })
        );
    }
}

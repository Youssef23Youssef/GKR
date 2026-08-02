use crate::field::F;
use ark_ff::Zero;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitError {
    NotEnoughLayers,

    EmptyLayer,

    InvalidRealWidth {
        layer: usize,
    },

    InvalidPaddedWidth {
        layer: usize,
    },

    InvalidIndexBits {
        layer: usize,
    },

    InvalidInputLayer,

    InvalidInputIndex {
        gate: usize,
        input_index: usize,
    },

    InvalidComputationGate {
        layer: usize,
        gate: usize,
    },

    InvalidChildIndex {
        layer: usize,
        gate: usize,
        child: usize,
    },

    WrongInputCount {
        expected: usize,
        actual: usize,
    },

    InvalidInputGateOrder {
        gate: usize,
        input_index: usize,
    },
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
/// - `outputs` contains the real values from the final layer.
///
/// These evaluated layer values will later become the GKR prover's witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CircuitEvaluation {
    layer_values: Vec<Vec<F>>,
    outputs: Vec<F>,
}

impl CircuitEvaluation {
    pub fn outputs(&self) -> &[F] {
        &self.outputs
    }

    pub fn layer_values(&self) -> &[Vec<F>] {
        &self.layer_values
    }
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

    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }

    pub fn real_width(&self) -> usize {
        self.real_width
    }

    pub fn padded_width(&self) -> usize {
        self.padded_width
    }

    pub fn index_bits(&self) -> usize {
        self.index_bits
    }

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
/// The final layer may contain one or more real output gates.
pub struct Circuit {
    layers: Vec<Layer>,
    expected_inputs: usize,
}

impl Circuit {
    pub fn new(layers: Vec<Layer>, expected_inputs: usize) -> Result<Self, CircuitError> {
        let circuit = Self {
            layers,
            expected_inputs,
        };

        circuit.validate()?;

        Ok(circuit)
    }

    pub fn validate(&self) -> Result<(), CircuitError> {
        // 1. Validate circuit shape:
        // Check that the circuit has at least two layers:
        if self.layers.len() < 2 {
            return Err(CircuitError::NotEnoughLayers);
        }

        // 2. Validate layer metadata:
        // - Every layer must contain at least one real gate, do not reference padded positions.
        // - `real_width` must match the number of stored gates, no computation layer is empty.
        // - `padded_width` must be the next power of two for `real_width`.
        // - `index_bits` must match `log2(padded_width)`.
        // - Padding exists only in evaluated value vectors, not as fake gates.
        for (layer_index, layer) in self.layers.iter().enumerate() {
            layer.validate_metadata(layer_index)?;
        }

        // 3. Validate input layer:
        // - Layer 0 must contain only input gates.
        // - Input layer width must match expected_inputs.
        // - Every input gate must reference an expected input index.
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

        // 5. Validate computation layers:
        // - Every gate after the input layer must be an Add or Mul gate.
        // - Child indices are interpreted as references to the immediately previous layer.
        // - Child indices must point to real gates, not padded positions.
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

    pub fn evaluate(&self, inputs: &[F]) -> Result<CircuitEvaluation, CircuitError> {
        self.validate()?;

        if inputs.len() != self.expected_inputs {
            return Err(CircuitError::WrongInputCount {
                expected: self.expected_inputs,
                actual: inputs.len(),
            });
        }

        // Allocate the evaluated value vector for the input layer using the padded layer width.
        // Real input gate values will be written into the first `real_width` positions
        // while any extra padded positions remain zero.
        //
        // This invariant is important for later GKR/MLE code:
        // `layer_values[i].len() == layers[i].padded_width`.
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

        // Evaluate each computation layer from inputs toward the output.
        //
        // Every computation layer gets its own padded value vector. Real gates are
        // evaluated from the immediately previous layer, while padded positions remain
        // zero. Saving each padded vector gives the prover the full layer-by-layer
        // witness needed later by GKR.
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

    // Test Circuit Validation

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

    // Test Circuit Evaluation

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

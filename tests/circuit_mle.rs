use ark_ff::Zero;
use gkr::{
    circuit::{Circuit, Gate, Layer},
    field::F,
    mle::evaluate_mle,
};

fn layer(gates: Vec<Gate>) -> Layer {
    Layer::new(gates).unwrap()
}

fn target_circuit() -> Circuit {
    Circuit::new(
        vec![
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
        4,
    )
    .unwrap()
}

#[test]
fn circuit_layer_mles_match_boolean_points() {
    let circuit = target_circuit();

    let inputs = vec![
        F::from(2u64),
        F::from(3u64),
        F::from(5u64),
        F::from(7u64),
    ];

    let evaluation = circuit.evaluate(&inputs).unwrap();
    let layer_values = evaluation.layer_values();

    // Layer 0 = [2, 3, 5, 7]
    // Little-endian indexing:
    // index 0 -> [0, 0]
    // index 1 -> [1, 0]
    // index 2 -> [0, 1]
    // index 3 -> [1, 1]
    assert_eq!(
        evaluate_mle(&layer_values[0], &[F::from(0u64), F::from(0u64)]).unwrap(),
        F::from(2u64)
    );
    assert_eq!(
        evaluate_mle(&layer_values[0], &[F::from(1u64), F::from(0u64)]).unwrap(),
        F::from(3u64)
    );
    assert_eq!(
        evaluate_mle(&layer_values[0], &[F::from(0u64), F::from(1u64)]).unwrap(),
        F::from(5u64)
    );
    assert_eq!(
        evaluate_mle(&layer_values[0], &[F::from(1u64), F::from(1u64)]).unwrap(),
        F::from(7u64)
    );

    // Layer 1 = [5, 12]
    assert_eq!(
        evaluate_mle(&layer_values[1], &[F::from(0u64)]).unwrap(),
        F::from(5u64)
    );
    assert_eq!(
        evaluate_mle(&layer_values[1], &[F::from(1u64)]).unwrap(),
        F::from(12u64)
    );

    // Layer 2 = [60], a constant 0-variable MLE.
    assert_eq!(evaluate_mle(&layer_values[2], &[]).unwrap(), F::from(60u64));
}

#[test]
fn padded_circuit_layer_mle_sees_padding_as_zero() {
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

    // Computation layer before padding is [3, 7, 11].
    // Padded layer table is [3, 7, 11, 0].
    assert_eq!(
        layer_values[1],
        vec![F::from(3u64), F::from(7u64), F::from(11u64), F::zero()]
    );

    // index 3 -> [1, 1], so the MLE should return the padded zero value.
    assert_eq!(
        evaluate_mle(&layer_values[1], &[F::from(1u64), F::from(1u64)]).unwrap(),
        F::zero()
    );
}

#[test]
fn circuit_layer_mle_evaluates_at_non_boolean_point() {
    let circuit = target_circuit();

    let inputs = vec![
        F::from(2u64),
        F::from(3u64),
        F::from(5u64),
        F::from(7u64),
    ];

    let evaluation = circuit.evaluate(&inputs).unwrap();
    let layer_values = evaluation.layer_values();

    // Layer 1 = [5, 12].
    //
    // For one variable:
    // V1(r) = 5 * (1 - r) + 12 * r.
    //
    // At r = 3:
    // V1(3) = 5 * (1 - 3) + 12 * 3
    //       = 5 * (-2) + 36
    //       = 26.
    let r = F::from(3u64);
    let expected = F::from(5u64) * (F::from(1u64) - r) + F::from(12u64) * r;

    assert_eq!(expected, F::from(26u64));
    assert_eq!(evaluate_mle(&layer_values[1], &[r]).unwrap(), expected);
}
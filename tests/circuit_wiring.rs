use gkr::{
    circuit::{Circuit, Gate, Layer},
    field::F,
    mle::evaluate_mle,
    wiring::{evaluate_add_predicate, evaluate_mul_predicate},
};

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
fn wiring_predicates_select_child_mles_for_add_gate() {
    let (layer0, layer1, layer2) = target_layers();
    let circuit = Circuit::new(vec![layer0.clone(), layer1.clone(), layer2], 4).unwrap();

    let inputs = vec![f(2), f(3), f(5), f(7)];
    let evaluation = circuit.evaluate(&inputs).unwrap();
    let layer_values = evaluation.layer_values();

    let g = [f(0)];
    let b = [f(0), f(0)];
    let c = [f(1), f(0)];

    let add = evaluate_add_predicate(&layer0, &layer1, &g, &b, &c).unwrap();
    let mul = evaluate_mul_predicate(&layer0, &layer1, &g, &b, &c).unwrap();

    let left = evaluate_mle(&layer_values[0], &b).unwrap();
    let right = evaluate_mle(&layer_values[0], &c).unwrap();
    let current = evaluate_mle(&layer_values[1], &g).unwrap();

    let reconstructed = add * (left + right) + mul * left * right;

    assert_eq!(add, f(1));
    assert_eq!(mul, f(0));
    assert_eq!(left, f(2));
    assert_eq!(right, f(3));
    assert_eq!(reconstructed, current);
    assert_eq!(current, f(5));
}

#[test]
fn wiring_predicates_select_child_mles_for_mul_gate() {
    let (layer0, layer1, layer2) = target_layers();
    let circuit = Circuit::new(vec![layer0, layer1.clone(), layer2.clone()], 4).unwrap();

    let inputs = vec![f(2), f(3), f(5), f(7)];
    let evaluation = circuit.evaluate(&inputs).unwrap();
    let layer_values = evaluation.layer_values();

    let g = [];
    let b = [f(0)];
    let c = [f(1)];

    let add = evaluate_add_predicate(&layer1, &layer2, &g, &b, &c).unwrap();
    let mul = evaluate_mul_predicate(&layer1, &layer2, &g, &b, &c).unwrap();

    let left = evaluate_mle(&layer_values[1], &b).unwrap();
    let right = evaluate_mle(&layer_values[1], &c).unwrap();
    let current = evaluate_mle(&layer_values[2], &g).unwrap();

    let reconstructed = add * (left + right) + mul * left * right;

    assert_eq!(add, f(0));
    assert_eq!(mul, f(1));
    assert_eq!(left, f(5));
    assert_eq!(right, f(12));
    assert_eq!(reconstructed, current);
    assert_eq!(current, f(60));
}

#[test]
fn wrong_wiring_has_zero_predicate_even_with_valid_layer_mles() {
    let (layer0, layer1, layer2) = target_layers();
    let circuit = Circuit::new(vec![layer0.clone(), layer1.clone(), layer2], 4).unwrap();

    let inputs = vec![f(2), f(3), f(5), f(7)];
    let evaluation = circuit.evaluate(&inputs).unwrap();
    let layer_values = evaluation.layer_values();

    let g = [f(0)];
    let wrong_b = [f(1), f(0)];
    let wrong_c = [f(0), f(0)];

    let add = evaluate_add_predicate(&layer0, &layer1, &g, &wrong_b, &wrong_c).unwrap();
    let mul = evaluate_mul_predicate(&layer0, &layer1, &g, &wrong_b, &wrong_c).unwrap();

    let left = evaluate_mle(&layer_values[0], &wrong_b).unwrap();
    let right = evaluate_mle(&layer_values[0], &wrong_c).unwrap();
    let contribution = add * (left + right) + mul * left * right;

    assert_eq!(add, f(0));
    assert_eq!(mul, f(0));
    assert_eq!(left, f(3));
    assert_eq!(right, f(2));
    assert_eq!(contribution, f(0));
}

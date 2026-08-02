# GKR Interactive Proof System

A Rust implementation of the Goldwasser–Kalai–Rothblum (GKR) interactive proof protocol for layered arithmetic circuits over finite fields.

## Overview

This repository is a ground-up implementation of the core components needed for the GKR protocol.

The project currently focuses on correctness, clarity, and explicit protocol structure before optimization. It builds the protocol in stages:

- finite-field arithmetic using Arkworks
- layered arithmetic circuit representation
- circuit validation and evaluation
- padded witness-table storage
- multilinear-extension utilities
- wiring predicates for circuit gates
- GKR layer-equation evaluation

The next major milestone is Sumcheck integration, followed by a full interactive GKR prover/verifier flow.

## Current scope

This repository does not yet implement the full GKR prover/verifier protocol.

The current implementation provides the foundation required for GKR:

```text
circuit evaluation
    -> padded layer witness tables
    -> multilinear-extension evaluation
    -> wiring predicates
    -> layer-equation evaluation
```

This is the bridge needed before implementing Sumcheck.

## Layer convention

Circuits are represented in input-to-output order:

```text
Layer 0      = input layer
Layer 1      = first computation layer
...
Last layer   = output layer
```

Circuit evaluation runs forward from the input layer to the output layer.

The GKR protocol later works in the opposite direction, reducing claims from the output layer back toward the input layer.

## Circuit invariants

- All circuit values are field elements.
- Layer 0 contains only input gates.
- Computation layers contain only addition and multiplication gates.
- Computation gates reference only the immediately preceding layer.
- Gates may reference only real positions, not padded positions.
- Padding positions evaluate to zero.
- The output layer may contain one or more real output gates.

## Multilinear-extension convention

Each evaluated layer is stored as a padded vector of field values. This vector is interpreted as the evaluation table of a multilinear polynomial over a Boolean hypercube.

This project uses little-endian Boolean indexing:

```text
index 0 -> [0, 0]
index 1 -> [1, 0]
index 2 -> [0, 1]
index 3 -> [1, 1]
```

For example, for:

```text
values = [2, 3, 5, 7]
```

we interpret the table as:

```text
V(0, 0) = 2
V(1, 0) = 3
V(0, 1) = 5
V(1, 1) = 7
```

The first Boolean variable is the least-significant bit of the vector index. This convention is also used for variable binding:

```text
bind x₀ = r:
new[j] = (1 - r) * old[2j] + r * old[2j + 1]
```

This indexing convention must remain consistent across MLE evaluation, wiring predicates, Sumcheck, and GKR.

## Wiring predicates and layer equation

For a transition from layer `i - 1` to layer `i`, the wiring predicates are:

```text
add_i(g, b, c)
mul_i(g, b, c)
```

where:

- `g` indexes a gate in the current layer,
- `b` indexes the left child in the previous layer,
- `c` indexes the right child in the previous layer.

The GKR layer equation is:

```text
F_i(g, b, c)
=
add_i(g, b, c) * (V_{i-1}(b) + V_{i-1}(c))
+
mul_i(g, b, c) * V_{i-1}(b) * V_{i-1}(c)
```

This polynomial is the object that Sumcheck will later prove over the Boolean hypercube.

## Initial target circuit

The first target circuit is:

```text
y = (a + b) * (c + d)
```

with layers:

```text
Layer 0: a, b, c, d
Layer 1: a + b, c + d
Layer 2: (a + b) * (c + d)
```

For inputs:

```text
a = 2
b = 3
c = 5
d = 7
```

the expected output is:

```text
y = 60
```

## Project layout

```text
src/
├── field.rs      # Field type selection and field-interface tests
├── circuit.rs    # Layered circuit model, validation, evaluation
├── mle.rs        # Multilinear-extension utilities
├── wiring.rs     # Wiring predicates and GKR layer equation
└── lib.rs        # Public module exports
```

Integration tests live under:

```text
tests/
├── circuit_mle.rs
└── circuit_wiring.rs
```

## Non-goals for the current milestone

- No Fiat–Shamir transform.
- No commitments.
- No proof serialization.
- No optimized prover.
- No generic circuit parser.
- No full GKR prover/verifier yet.

## Project goals

- Keep the implementation readable and protocol-faithful.
- Build each protocol component incrementally.
- Test each component independently before integrating it into the full protocol.
- Prefer correctness and clarity before optimization.
- Avoid hiding protocol steps behind premature abstractions.

## Roadmap

- [x] Finite-field integration
- [x] Layered circuit data model
- [x] Layer metadata and padding
- [x] Circuit validation
- [x] Circuit evaluation
- [x] Witness-table storage
- [x] Multilinear-extension utilities
- [x] Circuit/MLE integration tests
- [x] Wiring predicate MLEs
- [x] Layer-equation evaluation
- [x] Circuit/wiring integration tests
- [ ] Sumcheck protocol
- [ ] One-layer GKR reduction
- [ ] Full GKR prover
- [ ] Full GKR verifier
- [ ] Transcript support

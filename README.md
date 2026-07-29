# GKR Interactive Proof System

A Rust implementation of the Goldwasser-Kalai-Rothblum (GKR) protocol an interactive proof system.


## Overview

This repository is a ground-up implementation of the Goldwasser–Kalai–Rothblum (GKR) protocol.

The first milestone focuses on building the foundations required by the protocol:

- finite-field integration using Arkworks
- layered arithmetic circuit representation
- circuit validation
- circuit evaluation over field elements
- storage of all evaluated layer values as the prover witness

The implementation starts with a simple, explicit protocol model before moving toward multilinear extensions, Sumcheck integration, and the full GKR prover/verifier flow.

## Current scope

This repository currently implements the foundation for GKR, not the full protocol yet.

The first milestone is a validated layered circuit evaluator over a finite field.

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
- The final layer must contain exactly one real output gate.

## Initial target circuit

The first test circuit is:

```text
y = (a + b) * (c + d)
```

with layers:

```text
Layer 0: a, b, c, d
Layer 1: a + b, c + d
Layer 2: (a + b) * (c + d)
```

## Non-goals for the first milestone

- No Fiat–Shamir transform.
- No commitments.
- No proof serialization.
- No optimized prover.
- No generic circuit parser.
- No full GKR prover/verifier yet.

## Project goals

- Keep the implementation readable for others.
- Build each protocol component incrementally.
- Test every component independently before integrating it into the full protocol.
- Prefer correctness and clarity before optimization.

## Roadmap

- Finite-field integration ✔️
- Layered circuit data model ✔️
- Layer metadata and padding ✔️
- Circuit validation ✔️
- Circuit evaluation ✔️
- Witness storage
- Multilinear extension utilities
- Wiring predicate MLEs
- Sumcheck integration
- Transcript support
- GKR prover
- GKR verifier
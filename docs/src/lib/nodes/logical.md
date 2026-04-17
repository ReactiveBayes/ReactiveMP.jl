# [Logical operation nodes](@id lib-nodes-logical)

Logical nodes encode **hard Boolean constraints** between discrete binary variables. Each node represents a standard logic gate and enforces the corresponding truth table exactly — no approximation is needed for discrete variables.

These nodes are [`Deterministic`](@ref): they do not contribute probability mass directly, but constrain the joint distribution by making the outcome of the logic gate a deterministic function of its inputs.

## [Available operations](@id lib-nodes-logical-operations)

| Node | Operation | Output |
|------|-----------|--------|
| [`AND`](@ref) | Logical conjunction | `out = in1 ∧ in2` |
| [`OR`](@ref) | Logical disjunction | `out = in1 ∨ in2` |
| [`IMPLY`](@ref) | Logical implication | `out = in1 ⇒ in2` |
| [`NOT`](@ref) | Logical negation | `out = ¬in` |

All inputs and outputs are binary (Bernoulli-distributed) variables. The truth tables are:

**AND**

| `in1` | `in2` | `out` |
|-------|-------|-------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

**OR**

| `in1` | `in2` | `out` |
|-------|-------|-------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

**IMPLY** (`in1 ⇒ in2`, equivalent to `¬in1 ∨ in2`)

| `in1` | `in2` | `out` |
|-------|-------|-------|
| 0 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

**NOT**

| `in` | `out` |
|------|-------|
| 0 | 1 |
| 1 | 0 |

## [When to use logical nodes](@id lib-nodes-logical-when)

Logical nodes are useful whenever your model contains **prior structural knowledge** that relates binary events. Common use cases include:

- Encoding that "event A occurring implies event B also occurs": `b ~ IMPLY(a, b_evidence)`.
- Building fault-tree or diagnostic models where system failures are logical combinations of component failures.
- Expressing hard constraints in discrete Bayesian networks.

Because the constraints are exact, the resulting messages are also exact for binary inputs — no Monte Carlo or variational approximation is required.

```@docs
ReactiveMP.AND
ReactiveMP.OR
ReactiveMP.IMPLY
ReactiveMP.NOT
```

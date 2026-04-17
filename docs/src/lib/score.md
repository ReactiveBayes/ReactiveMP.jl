# [Score functions](@id lib-score)

ReactiveMP.jl computes the **Bethe free energy** as its variational objective during inference. The free energy decomposes into local contributions from each factor node and each variable node, which are accumulated reactively as messages update.

## [The Bethe free energy](@id lib-score-bethe)

The Bethe free energy approximates the negative log-evidence of the model:

```math
\mathcal{F}_{\text{Bethe}}[q] = \underbrace{\sum_f \langle -\log f \rangle_{q_f}}_{\text{average energy}} - \underbrace{\sum_f H[q_f]}_{\text{factor entropies}} + \underbrace{\sum_x (d_x - 1)\, H[q_x]}_{\text{variable entropies}}
```

where:
- The sum over `f` runs over all factor nodes, with `q_f` the local marginal over the factor's variables.
- The sum over `x` runs over all variable nodes, with `d_x` the degree (number of connected factors) and `q_x` the marginal of that variable.

ReactiveMP.jl computes each term reactively: whenever a marginal changes, the local contribution is recomputed and can be accumulated by subscribing to the score streams.

## [Score types](@id lib-score-types)

Three tag types are used to dispatch the `score` function:

| Type | Represents | Where used |
|------|-----------|-----------|
| `AverageEnergy` | `⟨-log f⟩_q` — the expected log-factor under the local marginal | Factor nodes |
| `DifferentialEntropy` | `-∫ q log q` — the Shannon entropy of a marginal | Factor and variable nodes |
| `FactorBoundFreeEnergy` | Local free energy contribution of one factor node | Factor nodes |
| `VariableBoundEntropy` | Scaled entropy contribution of one variable node | Variable nodes |

The full Bethe free energy is the sum of all `FactorBoundFreeEnergy` and `VariableBoundEntropy` scores across the graph.

## [The `score` function](@id lib-score-function)

`score` is the central dispatch point. It is called internally by the engine, but can also be called manually for inspection:

```julia
# Entropy of a marginal
score(DifferentialEntropy(), marginal)

# Average energy for a factor node
score(AverageEnergy(), MyNode, Val{(:x, :y)}(), (q_x, q_y), meta)
```

## [Defining average energy for custom nodes](@id lib-score-average-energy)

When adding a new factor node, the engine needs to know how to compute `⟨-log f⟩_q`. The `@average_energy` macro generates the required `score(::AverageEnergy, ...)` method:

```julia
@average_energy MyNode (q_x::NormalMeanVariance, q_y::Gamma) begin
    # return the average energy -⟨log f(x, y)⟩_{q(x)q(y)}
    mx, vx = mean_var(q_x)
    my     = mean(q_y)
    return 0.5 * log(2π) + 0.5 * (vx + mx^2) * my - ...
end
```

The macro handles argument naming, dispatch, and interface checking automatically. Marginals are named with a `q_` prefix matching the node interface names declared in the corresponding `@node` definition.

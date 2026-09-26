# [BIFM](@id page-bifm)

## Overview

`BIFM` is one time slice of a linear state-space model with known matrices, as a single
deterministic node: the previous state and an input make the next state, which makes the output.
A chain of them, started by [`BIFMHelper`](@ref) and ended by a flat prior on the last state, is
smoothed in two passes. The backward pass sends information-form messages towards the earlier
states; at the start of the chain [`BIFMHelper`](@ref) turns it into the forward pass, whose
messages are the marginals themselves. On a chain the result is the Rauch–Tung–Striebel
smoother's.

## Definition

```math
z_{next} = A z_{prev} + B\, u, \qquad out = C z_{next}
```

The backward message towards `zprev`, from the messages on `out`, `in` and `znext`, in
weighted-mean and precision form:

```math
\begin{aligned}
\xi_z &= C^\top \xi_{out} + \xi_{znext}, &
\Lambda_z &= C^\top \Lambda_{out} C + \Lambda_{znext}, \\
H &= (\Lambda_{in} + B^\top \Lambda_z B)^{-1}, &
\tilde\xi_z &= \xi_z + \Lambda_z B H (-\xi_{in} - B^\top \xi_z), \\
\tilde\Lambda_z &= \Lambda_z (I - B H B^\top \Lambda_z), &
m_{zprev} &= \mathcal{N}^{-1}\big(A^\top \tilde\xi_z,\; A^\top \tilde\Lambda_z A\big).
\end{aligned}
```

The forward rules take the marginal of `zprev`, `N(μ, Σ)`, and recompute these quantities from
the messages on their own edges. With `F = I - Λ_z B H Bᵀ`, the marginal of `znext` is
`N(Fᵀ A μ + B (Σ_in Bᵀ ξ̃_z + μ_in), Fᵀ A Σ Aᵀ F + B H Bᵀ)` and the marginal of `out` its image
under `C`; the marginal of `in` follows Wadehn (2016), *On sparsity by NUV-EM*, in the dual
parameterisation.

## Interfaces

| name | meaning | messages the rules take |
|---|---|---|
| `out` | the output, `C znext` | `MvNormal` |
| `in` | the input `u` | `MvNormal` |
| `zprev` | the previous state | `TerminalProdArgument` of an `MvNormal`, the forward pass's marginal |
| `znext` | the next state | `MvNormal`, possibly with zero precision at the end of the chain |

Towards `zprev` the rule sends an `MvNormal` in weighted-mean and precision form; towards `in`,
`out` and `znext` a `TerminalProdArgument` of an `MvNormal`, which the engine takes as the
marginal without a product.

## Algorithm

```@docs
BIFMSmoother
```

The node declares no algorithm of its own, so a model gives every `BIFM` node a `BIFMSmoother`;
under the default one, [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm),
no rule is found. The forward rules read the message on their own edge, besides the messages the
default scheme gives them, and keep only working memory between the steps of one call.

## Supported rules

```@example
using MessagePassingRulesBase, BIFMMessagePassingRules # hide
MessagePassingRulesBase.rule_coverage(BIFM)
```

Every message rule is belief propagation on the messages. The marginal rule for
`q(in, zprev, znext)`, the joint that the free energy of a deterministic node reads, exists only
to throw a [`BIFMFreeEnergyError`](@ref BIFMMessagePassingRules.BIFMFreeEnergyError): there is no
average energy, and no free energy.

## Example

The backward message of a scalar random walk observed directly: the message on `out` is
`N(1, 1)`, the input's `N(0, 1)`, and nothing is known of the next state, so the message on
`zprev` is `N(1, 2)`.

```jldoctest
julia> using BIFMMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily

julia> result = @call_message_update_rule(
           node = BIFM, target = :zprev, algorithm = BIFMSmoother([1.0;;], [1.0;;], [1.0;;]),
           m = (
               out = MvNormalMeanPrecision([1.0], [1.0;;]),
               in = MvNormalMeanPrecision([0.0], [1.0;;]),
               znext = MvNormalWeightedMeanPrecision([0.0], [0.0;;]),
           ),
       );

julia> m, V = mean_cov(getresult(result));

julia> m ≈ [1.0] && V ≈ [2.0;;]
true
```

## Limitations

- **No free energy**: the joint marginal of the inputs and the average energy of
  [`BIFMHelper`](@ref) throw a
  [`BIFMFreeEnergyError`](@ref BIFMMessagePassingRules.BIFMFreeEnergyError).
- **Multivariate normal messages only**, and the forward rules need the marginal of `zprev` as a
  `TerminalProdArgument`, so the chain must start with [`BIFMHelper`](@ref).
- The matrices are known: `BIFMSmoother` carries them, and nothing learns them.

## API

The node, and the error that asking for its free energy throws.

```@docs
BIFM
BIFMMessagePassingRules.BIFMFreeEnergyError
```

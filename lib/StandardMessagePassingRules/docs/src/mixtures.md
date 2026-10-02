# Mixtures

A mixture node models a variable `out` drawn from one of several components, with a one-hot
`switch` that selects the component. Models use it for clustering and regime switching. The message
towards `switch` says which component explains `out` best, and the message towards a component
treats `out` as that component's data, weighted by the probability of the switch.

The page has three mixture nodes. Unlike the distribution nodes, each declares an
[algorithm](@extref MessagePassingRulesBase glossary-algorithm) of its own, because its rules
ignore the [factorisation](@extref MessagePassingRulesBase glossary-factorisation).
[`NormalMixture`](@ref) and [`GammaMixture`](@ref) are always
[variational](@extref MessagePassingRulesBase glossary-vmp): their rules take
[marginals](@extref MessagePassingRulesBase glossary-marginal) and use expectations such as
`E[log p]`. [`Mixture`](@ref) is always
[belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation): its rules take
[messages](@extref MessagePassingRulesBase glossary-message) and their
[log scales](@extref MessagePassingRulesBase glossary-log-scale), which weigh the components by
their evidence. The algorithm is the node's default, so a model need not name it.

| node | components | algorithm | rules | average energy |
|---|---|---|---|---|
| [`NormalMixture`](@ref) | normals with means `m` and precisions `p` | [`NormalMixtureVMP`](@ref) | variational, over marginals | yes |
| [`GammaMixture`](@ref) | Gammas with shapes `a` and rates `b` | [`GammaMixtureVMP`](@ref) | variational, over marginals | yes |
| [`Mixture`](@ref) | any, the messages of `inputs` | [`MixtureBP`](@ref) | belief propagation, over messages and their log scales | no |

The components are [groups](@extref MessagePassingRulesBase glossary-group), declared with `...`
(`:m...`). The number of components is the number of members a model connects, at least two.
`NormalMixture` and `GammaMixture` need the
[mean-field](@extref MessagePassingRulesBase glossary-mean-field) factorisation, and as many
members in one group as in the other. The engine's `factornode` checks both.

```@setup mixtures
using MessagePassingRulesBase, StandardMessagePassingRules, ExponentialFamily, Distributions, BayesBase
```

## Example

With both components known, the switch probabilities of a point near the first component's mean
favour that component:

```jldoctest mixtures
julia> using StandardMessagePassingRules, MessagePassingRulesBase, BayesBase, Distributions

julia> switch = getresult(@call_message_update_rule(
           node = NormalMixture, target = :switch,
           q = (out = PointMass(0.5), m = (PointMass(0.0), PointMass(4.0)), p = (PointMass(1.0), PointMass(1.0))),
       ));

julia> probvec(switch)[1] > 0.99
true
```

## NormalMixture

```math
p(\mathrm{out} \mid \mathrm{switch}, m, p) = \prod_{k=1}^K \mathcal{N}(\mathrm{out} \mid m_k, p_k^{-1})^{\mathrm{switch}_k}
```

```@example mixtures
MessagePassingRulesBase.nodespec(NormalMixture)
```

The declaration draws the two groups, `m` for the means and `p` for the precisions. It also lists
the requirements that the engine checks: matched groups, at least two members and the mean-field
factorisation.

```@example mixtures
MessagePassingRulesBase.rule_coverage(NormalMixture)
```

The message towards `switch` compares the expected log-density of `out` under each component:

```@example mixtures
@call_message_update_rule(
    node = NormalMixture, target = :switch,
    q = (out = PointMass(0.5), m = (PointMass(0.0), PointMass(4.0)), p = (PointMass(1.0), PointMass(1.0))),
)
```

The message towards the first mean, `(:m, 1)`, is a normal around `E[out]`. Its precision is
`E[p₁]`, weighted by the switch's probability of the first component, `0.9 × 2`. The rule reads
only the first precision, `q[:p][k]`, so the call passes `nothing` for the second, as a graph
does:

```@example mixtures
@call_message_update_rule(
    node = NormalMixture, target = (:m, 1),
    q = (out = PointMass(0.5), switch = Categorical([0.9, 0.1]), p = (GammaShapeRate(2.0, 1.0), nothing)),
)
```

The two rules towards each component are the variational one and one for an integer `PointMass`
switch, which throws an error.

```@docs
NormalMixture
GaussianMixture
NormalMixtureVMP
```

## GammaMixture

```math
p(\mathrm{out} \mid \mathrm{switch}, a, b) = \prod_{k=1}^K \mathrm{Gamma}(\mathrm{out} \mid a_k, b_k)^{\mathrm{switch}_k},
```

with shapes `a[k]` and rates `b[k]`, as in `GammaShapeRate`.

```@example mixtures
MessagePassingRulesBase.rule_coverage(GammaMixture)
```

As for `NormalMixture`, the second rule towards each component throws the error for an integer
switch. The rules towards `out`, `switch` and the shapes `a` need a Gamma marginal of each rate
`b[k]`, and they refuse a `PointMass` rate.

```@docs
GammaMixture
GammaMixtureVMP
```

## Mixture

```math
p(\mathrm{out} \mid \mathrm{switch}, \mathrm{inputs}) = \prod_{k=1}^K \mathrm{inputs}_k(\mathrm{out})^{\mathrm{switch}_k}
```

```@example mixtures
MessagePassingRulesBase.nodespec(Mixture)
```

```@example mixtures
MessagePassingRulesBase.rule_coverage(Mixture)
```

The rules are declared on `MixtureBP` itself, so that they apply for every product strategy,
the default `MixtureBP{GenericProd}` included.

The message towards `switch` weighs each component by the evidence of `out` under it: the log
scale of the product of the message on `out` and the component's message, plus the incoming log
scales. A call passes the incoming log scales with the `logscale` keyword:

```@example mixtures
@call_message_update_rule(
    node = Mixture, target = :switch,
    m = (out = NormalMeanVariance(0.5, 0.1), inputs = (NormalMeanVariance(0.0, 1.0), NormalMeanVariance(4.0, 1.0))),
    logscale = (out = 0.0, inputs = (0.0, 0.0)),
)
```

**Limitations.** `Mixture` has no
[average energy](@extref MessagePassingRulesBase glossary-average-energy), so the free energy of
a model with a `Mixture` is an error. The graph must track log scales, and an unknown incoming
log scale is an error.

```@docs
Mixture
MixtureBP
```

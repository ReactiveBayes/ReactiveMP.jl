# Mixtures

Three mixture nodes, where a one-hot `switch` selects one of several components for `out`. Unlike
the distribution nodes, each declares an algorithm of its own, because its rules ignore the
factorisation: [`NormalMixture`](@ref) and [`GammaMixture`](@ref) are always variational,
[`Mixture`](@ref) always belief propagation. The algorithm is the node's default, so a model
need not name it.

| node | components | algorithm | rules | average energy |
|---|---|---|---|---|
| [`NormalMixture`](@ref) | normals with means `m` and precisions `p` | [`NormalMixtureVMP`](@ref) | variational, over marginals | yes |
| [`GammaMixture`](@ref) | Gammas with shapes `a` and rates `b` | [`GammaMixtureVMP`](@ref) | variational, over marginals | yes |
| [`Mixture`](@ref) | any, the messages of `inputs` | [`MixtureBP`](@ref) | belief propagation, over messages and their log scales | no |

The component groups are declared with `...` (`:m...`), so the number of components is how many
members a model connects, at least two. `NormalMixture` and `GammaMixture` need a mean-field
factorisation and as many members in one group as in the other; `factornode` checks both.

```@setup mixtures
using MessagePassingRulesBase, StandardMessagePassingRules
```

## Example

With both components known, the responsibilities of a point near the first component's mean
favour it:

```jldoctest mixtures
julia> using StandardMessagePassingRules, MessagePassingRulesBase, BayesBase, Distributions

julia> switch = getresult(@call_message_update_rule(
           node = NormalMixture, target = :switch,
           q = (out = PointMass(0.5), m = (PointMass(0.0), PointMass(4.0)), p = (PointMass(1.0), PointMass(1.0))),
       ));

julia> probvec(switch)[1] > 0.99
true
```

```julia
@model function gaussian_mixture(y)
    s ~ Dirichlet([1.0, 1.0])
    m[1] ~ NormalMeanVariance(-2.0, 10.0)
    m[2] ~ NormalMeanVariance(2.0, 10.0)
    p[1] ~ GammaShapeRate(1.0, 1.0)
    p[2] ~ GammaShapeRate(1.0, 1.0)
    for i in eachindex(y)
        z[i] ~ Categorical(s)
        y[i] ~ NormalMixture(switch = z[i], m = m, p = p)
    end
end
```

## NormalMixture

```@example mixtures
MessagePassingRulesBase.rule_coverage(NormalMixture)
```

The two rules towards each component are the variational one and the one for an integer
`PointMass` switch, which is an error.

```@docs
NormalMixture
GaussianMixture
NormalMixtureVMP
```

## GammaMixture

```@example mixtures
MessagePassingRulesBase.rule_coverage(GammaMixture)
```

As for `NormalMixture`, the second rule towards each component is the error for an integer
switch.

```@docs
GammaMixture
GammaMixtureVMP
```

## Mixture

```@example mixtures
MessagePassingRulesBase.rule_coverage(Mixture)
```

The table has two `MixtureBP` columns: the first is the type of the node's default algorithm,
`MixtureBP{GenericProd}`, the second `MixtureBP` itself, on which the rules are declared so that
they apply for every product strategy.

**Limitations.** No average energy, so the free energy of a model with a `Mixture` is an error;
the graph must track log scales, and an unknown incoming log scale is an error.

```@docs
Mixture
MixtureBP
```

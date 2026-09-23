"""
    NormalMixture

A mixture of univariate normals: `out` is drawn from the component `switch` selects, with
means `m` and precisions `p`, both groups with one member per component. v6 called it
`NormalMixture{N}`; the number of components is now the groups' length, not a type
parameter.

Its rules run under its own algorithm, [`NormalMixtureVMP`](@ref), and are always variational:
they consume marginals only, whatever the factorisation. The multivariate form, with
`MvNormalMeanPrecision` components, arrives when those rules are ported.

The `:out` and `:switch` dependencies list the precisions before the means. An engine
subscribes to a target's inputs in the order they are declared, and in variational message
passing that order is the update schedule: each component's precision is updated before its
mean. Both orders reach the same optimum; on the Phase 4.5 mixture model, precisions first
converges in about 7 iterations and means first in about 13 (`DISCUSSION.md` §3.24). The
order of the members within a group changes nothing, since they do not depend on each other.
"""
struct NormalMixture end

"""
    NormalMixtureVMP()

[`NormalMixture`](@ref)'s own algorithm: variational message passing, **regardless of the
factorisation**. Its rules take the marginals of `out`, `switch` and the components, and never
their messages, which is what v6's mixture did too. It stands alone and inherits nothing from
`DefaultAlgorithm`, which has no mixture rules.
"""
struct NormalMixtureVMP <: AbstractAlgorithm end

@define_factor_node(
    node = NormalMixture,
    type = Stochastic,
    interfaces = [:out, :switch, :m..., :p...],
    algorithm = NormalMixtureVMP,
    dependencies = [
        :out => (q[:switch], q[:p...], q[:m...]),
        :switch => (q[:out], q[:p...], q[:m...]),
        (:m, k) => (q[:out], q[:switch], q[:p][k]),
        (:p, k) => (q[:out], q[:switch], q[:m][k]),
    ],
)

@define_average_energy(
    node = NormalMixture,
    args = (q[:out]::Any, q[:switch]::Any, q[:m...]::Any, q[:p...]::Any),
    body = (args) -> begin
        z = probvec(args.q[:switch])
        sum(k -> z[k] * normal_mean_precision_energy(args.q[:out], args.q[:m][k], args.q[:p][k]), eachindex(args.q[:m]))
    end,
)

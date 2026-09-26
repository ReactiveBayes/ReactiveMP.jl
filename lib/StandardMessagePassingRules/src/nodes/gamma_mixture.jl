"""
    GammaMixture

A mixture of Gamma distributions: `out` is drawn from the component the one-hot `switch`
selects, a `GammaShapeRate` with shape `a` and rate `b`, both groups with one member per
component,

```math
p(\\mathrm{out} \\mid \\mathrm{switch}, a, b) = \\prod_{k=1}^K
\\mathrm{Gamma}(\\mathrm{out} \\mid a_k, b_k)^{\\mathrm{switch}_k}.
```

The number of components `K` is the groups' length. Its interfaces are `out`, `switch` and the
groups `a` and `b`.

$(DOC_MIXTURE_REQUIREMENTS) The rates' marginals must be Gamma distributions.

Its rules run under its own algorithm, [`GammaMixtureVMP`](@ref), and are always variational.
Towards `out` the message is a `GammaShapeRate`, towards `switch` a `Categorical`, towards
`(:a, k)` a [`GammaShapeLikelihood`](@ref) and towards `(:b, k)` a `GammaShapeRate`. It has an
average energy, each component's weighted by the switch. The `:out` and `:switch` dependencies
list the rates before the shapes, which is the update schedule; the order within a group
changes nothing.
"""
struct GammaMixture end

"""
    GammaMixtureVMP()

[`GammaMixture`](@ref)'s own algorithm. $(DOC_MIXTURE_VMP)
"""
struct GammaMixtureVMP <: AbstractAlgorithm end

@define_factor_node(
    node = GammaMixture,
    type = Stochastic,
    interfaces = [:out, :switch, :a..., :b...],
    algorithm = GammaMixtureVMP,
    matched_groups = [(:a, :b)],
    min_group_length = 2,
    factorisation = :meanfield,
    dependencies = [
        :out => (q[:switch], q[:b...], q[:a...]),
        :switch => (q[:out], q[:b...], q[:a...]),
        (:a, k) => (q[:out], q[:switch], q[:b][k]),
        (:b, k) => (q[:out], q[:switch], q[:a][k]),
    ],
)

# Each component's GammaShapeRate energy, weighted by the switch.
@define_average_energy(
    node = GammaMixture,
    args = (q[:out]::Any, q[:switch]::Any, q[:a...]::Any, q[:b...]::GammaDistributionsFamily),
    body = (args) -> begin
        z = probvec(args.q[:switch])
        sum(k -> z[k] * gamma_shape_rate_energy(args.q[:out], args.q[:a][k], args.q[:b][k]), eachindex(args.q[:a]))
    end,
)

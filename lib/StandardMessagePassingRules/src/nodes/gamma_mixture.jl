"""
    GammaMixture

A mixture of Gamma distributions: `out` is drawn from the component `switch` selects, a
`GammaShapeRate` with shape `a` and rate `b`, both groups with one member per component. v6
called it `GammaMixture{N}`; the number of components is now the groups' length. A node needs
at least two components, as many shapes as rates, and a mean-field factorisation.

Its rules run under its own algorithm, [`GammaMixtureVMP`](@ref), and are always variational.
The `:out` and `:switch` dependencies list the rates before the shapes, which is the update
schedule, as v6 had it; the order within a group changes nothing.
"""
struct GammaMixture end

"""
    GammaMixtureVMP()

[`GammaMixture`](@ref)'s own algorithm: variational message passing, **regardless of the
factorisation**. Its rules take the marginals of `out`, `switch` and the components.
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

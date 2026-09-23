"""
    NormalMixture

A mixture of univariate normals: `out` is drawn from the component `switch` selects, with
means `m` and precisions `p`, both groups with one member per component. v6 called it
`NormalMixture{N}`; the number of components is now the groups' length, not a type
parameter.

Its rules run under `VMP` and require a mean-field factorisation, as in v6. The
multivariate form, with `MvNormalMeanPrecision` components, arrives when those rules are
ported.
"""
struct NormalMixture end

@define_factor_node(
    node = NormalMixture,
    type = Stochastic,
    interfaces = [:out, :switch, :m..., :p...],
    algorithm = VMP,
    dependencies = [
        :out => (q[:switch], q[:m...], q[:p...]),
        :switch => (q[:out], q[:m...], q[:p...]),
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

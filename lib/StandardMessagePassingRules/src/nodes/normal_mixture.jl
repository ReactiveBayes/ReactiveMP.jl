# What the two variational mixtures share, NormalMixture and GammaMixture.
const DOC_MIXTURE_REQUIREMENTS = rstrip(
    """
    A node needs at least two components, as many members in one group as in the other, and a
    mean-field factorisation; `factornode` checks all three. The switch's marginal must be a
    `Categorical`, a `Bernoulli` or a one-hot vector `PointMass`: an integer `PointMass` switch
    is an error, which says to one-hot encode it.
    """
)

const DOC_MIXTURE_VMP = rstrip(
    """
    Variational message passing, **regardless of the factorisation**: every rule takes the
    marginals of `out`, `switch` and the components, never their messages. It is the node's
    default, so a model need not name it. It stands alone, a direct subtype of
    [`AbstractAlgorithm`](@extref MessagePassingRulesBase.AbstractAlgorithm), and inherits
    nothing from [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), which has
    no mixture rules.
    """
)

"""
    NormalMixture

A mixture of normals: `out` is drawn from the component the one-hot `switch` selects, with means
`m` and precisions `p`, both groups with one member per component,

```math
p(\\mathrm{out} \\mid \\mathrm{switch}, m, p) = \\prod_{k=1}^K
\\mathcal{N}(\\mathrm{out} \\mid m_k, p_k^{-1})^{\\mathrm{switch}_k}.
```

A component is a `NormalMeanPrecision`, with a Gamma precision, or, when `out` is a vector, an
`MvNormalMeanPrecision`, with a Wishart precision. The number of components `K` is the groups'
length, not a type parameter. Its interfaces are `out`, `switch` and the groups `m` and `p`;
[`GaussianMixture`](@ref) is an alias.

$(DOC_MIXTURE_REQUIREMENTS)

Its rules run under its own algorithm, [`NormalMixtureVMP`](@ref), and are always variational.
Towards `out` the message is a normal in weighted-mean form, towards `switch` a `Categorical`
of the components' responsibilities, towards `(:m, k)` a normal and towards `(:p, k)` a
`GammaShapeRate` or a Wishart likelihood. It has an average energy, each component's weighted
by the switch.

The `:out` and `:switch` dependencies list the precisions before the means. An engine
subscribes to a target's inputs in the order they are declared, and in variational message
passing that order is the update schedule: each component's precision is updated before its
mean. Both orders reach the same optimum, but precisions first converges faster, in about 7
iterations against about 13 on a measured mixture model. The order of the members within a
group changes nothing, since they do not depend on each other.

# Examples

```jldoctest; setup = :(using StandardMessagePassingRules, MessagePassingRulesBase, BayesBase, Distributions)
julia> switch = getresult(@call_message_update_rule(
           node = NormalMixture, target = :switch,
           q = (out = PointMass(0.0), m = (PointMass(0.0), PointMass(5.0)), p = (PointMass(1.0), PointMass(1.0))),
       ));

julia> probvec(switch)[1] > 0.99
true
```
"""
struct NormalMixture end

"""
    GaussianMixture

An alias of [`NormalMixture`](@ref).
"""
const GaussianMixture = NormalMixture

"""
    NormalMixtureVMP()

[`NormalMixture`](@ref)'s own algorithm. $(DOC_MIXTURE_VMP)
"""
struct NormalMixtureVMP <: AbstractAlgorithm end

@define_factor_node(
    node = NormalMixture,
    type = Stochastic,
    interfaces = [:out, :switch, :m..., :p...],
    algorithm = NormalMixtureVMP,
    matched_groups = [(:m, :p)],
    min_group_length = 2,
    factorisation = :meanfield,
    dependencies = [
        :out => (q[:switch], q[:p...], q[:m...]),
        :switch => (q[:out], q[:p...], q[:m...]),
        (:m, k) => (q[:out], q[:switch], q[:p][k]),
        (:p, k) => (q[:out], q[:switch], q[:m][k]),
    ],
)

# A component's energy: its own NormalMeanPrecision node's, or MvNormalMeanPrecision's when
# `out` is a vector.
mixture_component_energy(q_out, q_m, q_p) = mixture_component_energy(variate_form(typeof(q_out)), q_out, q_m, q_p)
mixture_component_energy(::Type{Univariate}, q_out, q_m, q_p) = normal_mean_precision_energy(q_out, q_m, q_p)
mixture_component_energy(::Type{Multivariate}, q_out, q_m, q_p) =
    mv_normal_mean_precision_energy(ndims(q_out), q_p, difference_moment(q_out, q_m))

@define_average_energy(
    node = NormalMixture,
    args = (q[:out]::Any, q[:switch]::Any, q[:m...]::Any, q[:p...]::Any),
    body = (args) -> begin
        z = probvec(args.q[:switch])
        sum(k -> z[k] * mixture_component_energy(args.q[:out], args.q[:m][k], args.q[:p][k]), eachindex(args.q[:m]))
    end,
)

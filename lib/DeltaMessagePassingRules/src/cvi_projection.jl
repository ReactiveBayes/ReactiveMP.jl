# `CVIProjection`, the Delta node's projection method. Its rules need ExponentialFamilyProjection
# and live in the package extension `DeltaMessagePassingRulesProjectionExt`; without it the
# method is not accepted, and the error says which package to load.

"""
    CVISamplingStrategy

How [`CVIProjection`](@ref) samples the other inputs while it projects one input of a node with
several: [`FullSampling`](@ref) or [`MeanBased`](@ref). A node with one input does not sample for
its joint.
"""
abstract type CVISamplingStrategy end

"""
    FullSampling(samples::Int = 10)

The [`CVISamplingStrategy`](@ref) that draws `samples` samples from each input's distribution,
the proposal's or, before there is one, the input's message. The default strategy of
[`CVIProjection`](@ref).
"""
struct FullSampling <: CVISamplingStrategy
    samples::Int
    FullSampling(samples::Int = 10) = new(samples)
end

"""
    MeanBased()

The [`CVISamplingStrategy`](@ref) that uses each input's mean as its only sample: much faster
than [`FullSampling`](@ref), and less accurate for a strongly nonlinear function or a wide
distribution.
"""
struct MeanBased <: CVISamplingStrategy end

"""
    ProposalDistributionContainer(distribution)
    ProposalDistributionContainer{Any}(nothing)

A mutable holder of the proposal [`CVIProjection`](@ref) samples the inputs from, for a node with
several inputs. `distribution` is `nothing`, and the inputs' messages are sampled, or a
`FactorizedJoint` of the inputs, whose components are sampled. The rule for the joint over the
inputs replaces it with its result on every call.
"""
mutable struct ProposalDistributionContainer{PD}
    distribution::PD
end

"""
    CVIProjection(; outsamples = 100, out_prjparams = nothing, in_prjparams = nothing,
                    proposal_distribution = ProposalDistributionContainer{Any}(nothing),
                    sampling_strategy = FullSampling(10))

The Delta node's projection method, for [`DeltaApproximation`](@ref)`(method = CVIProjection())`:
it samples the inputs, pushes the samples through the function, and projects the result onto an
exponential family with ExponentialFamilyProjection. It takes messages of any family the
projection can target, where [`Unscented`](@extref MessagePassingRulesApproximations.Unscented)
and [`Linearization`](@extref MessagePassingRulesApproximations.Linearization) take normals only.

Its rules are in the package extension `DeltaMessagePassingRulesProjectionExt`, loaded with
`using ExponentialFamilyProjection`; until then `DeltaApproximation` does not accept the method,
and its error says which package to load.

# Keywords

- `outsamples`: the number of samples drawn from each input's marginal for the message towards
  `out`. Default `100`.
- `out_prjparams`: the family of the message towards `out`. `nothing`, the default, projects
  onto the family of `q(out)` with the default projection parameters; a `ProjectedTo` is used as
  it is; `ProjectionParameters` project onto the family of `q(out)` with those parameters.
- `in_prjparams`: the families of the inputs' marginals, a `NamedTuple` keyed `in_1`, `in_2`, …,
  each as for `out_prjparams`, or `nothing`, the default, for every input. A missing key, or
  `nothing`, projects input `k` onto the family of its message.
- `proposal_distribution`: the [`ProposalDistributionContainer`](@ref) the inputs are sampled
  from, for a node with several inputs. Default: an empty one, so the first call samples the
  inputs' messages.
- `sampling_strategy`: how those inputs are sampled, [`FullSampling`](@ref) or
  [`MeanBased`](@ref). Default [`FullSampling`](@ref)`(10)`.

# The rules

- Towards `out`: samples of the joint over the inputs, `q[(:in,)]`, a `FactorizedJoint`, pushed
  through the function and projected; the message is that projection divided by the message
  that arrived from `out`, left unevaluated, so its product with that message is the projection.
- The joint over the inputs, from `m[:out]` and the inputs' messages: for one input, its message
  reweighted by the likelihood of `out` and projected; for several, each input in turn,
  against samples of the others' latest projections. The result becomes the next proposal.
- Towards `(:in, k)`: the input's component of the joint divided by its message, left
  unevaluated.

The samples are drawn from the rule context's generator, `ctx.rng`, which the engine owns. The
joint rule writes its result into `proposal_distribution`, so the method carries state, its rule
is impure, and one `CVIProjection` value given to several nodes shares one proposal among them.
A known inverse is not used, and is ignored with a warning.

# Examples

```julia
using DeltaMessagePassingRules, ExponentialFamilyProjection

algorithm = DeltaApproximation(method = CVIProjection(sampling_strategy = MeanBased()))
```
"""
Base.@kwdef struct CVIProjection{S, OF, IFS, PD, SS} <: MessagePassingRulesApproximations.AbstractApproximationMethod
    outsamples::S = 100
    out_prjparams::OF = nothing
    in_prjparams::IFS = nothing
    proposal_distribution::PD = ProposalDistributionContainer{Any}(nothing)
    sampling_strategy::SS = FullSampling(10)
end

"""
    get_kth_in_form(method::CVIProjection, k::Int)

The projection family `method` names for input `k`, its `in_prjparams.in_k`, or `nothing` when
it names none.
"""
get_kth_in_form(::CVIProjection{S, OF, Nothing}, ::Int) where {S, OF} = nothing
get_kth_in_form(method::CVIProjection, k::Int) = get(method.in_prjparams, Symbol("in_", k), nothing)

delta_method_hint(::CVIProjection) = "`CVIProjection`'s rules are in an extension: load it with `using ExponentialFamilyProjection`."

function check_inverse(::CVIProjection, inverse)
    inverse === nothing || @warn "`CVIProjection` does not use a known inverse; the one given is ignored."
    return nothing
end

# Towards `out`, the projection reads the message on its own edge, the marginal of `out` and the
# joint over the inputs; towards an input, it divides the joint by the input's own message.
@define_dependencies(
    node = DeltaFn, algorithm = DeltaApproximation{<:CVIProjection, Nothing},
    dependencies = [:out => (m[:out], q[:out], q[(:in,)]), (:in, k) => (m[:in][k], q[(:in,)])],
    free_energy_partition = [(:out,), (:in,)],
)

@define_dependencies(
    node = DeltaFn, algorithm = DeltaApproximation{<:CVIProjection, <:Union{Function, Tuple{Vararg{Function}}}},
    dependencies = [:out => (m[:out], q[:out], q[(:in,)]), (:in, k) => (m[:in][k], q[(:in,)])],
    free_energy_partition = [(:out,), (:in,)],
)

# `CVIProjection`, the Delta node's projection method. Its rules need ExponentialFamilyProjection
# and live in the package extension `DeltaMessagePassingRulesProjectionExt`; without it the
# method is not accepted, and the error says which package to load.

"""
    CVISamplingStrategy

How [`CVIProjection`](@ref) samples the inputs when it projects their joint: [`FullSampling`](@ref)
or [`MeanBased`](@ref).
"""
abstract type CVISamplingStrategy end

"""
    FullSampling(samples = 10)

Draw `samples` samples from each input's distribution.
"""
struct FullSampling <: CVISamplingStrategy
    samples::Int
    FullSampling(samples::Int = 10) = new(samples)
end

"""
    MeanBased()

Use each input's mean as its only sample: much faster than [`FullSampling`](@ref), and less
accurate for a nonlinear function or a distribution far from its mean.
"""
struct MeanBased <: CVISamplingStrategy end

"""
    ProposalDistributionContainer(distribution)

The proposal [`CVIProjection`](@ref) samples the inputs from, which the joint rule replaces with
its result on every call; `nothing` samples the incoming messages instead.
"""
mutable struct ProposalDistributionContainer{PD}
    distribution::PD
end

"""
    CVIProjection(; outsamples = 100, out_prjparams = nothing, in_prjparams = nothing,
                    proposal_distribution = ProposalDistributionContainer{Any}(nothing),
                    sampling_strategy = FullSampling(10))

The Delta node's projection method: it samples the inputs, pushes the samples through the
function, and projects onto an exponential family with ExponentialFamilyProjection, which must
be loaded (`using ExponentialFamilyProjection`) for the node to accept it.

- `outsamples`: the number of samples for the message towards `out`;
- `out_prjparams`: the family of the message towards `out`, a `ProjectedTo`, or projection
  parameters, or `nothing` for the family of `q(out)`;
- `in_prjparams`: a named tuple of the families of the inputs, keyed `in_k`, each as for `out`;
- `proposal_distribution`: the proposal to sample the inputs from, updated by every call of the
  joint rule, so the method carries state and its rules are impure;
- `sampling_strategy`: [`FullSampling`](@ref) or [`MeanBased`](@ref).

The samples are drawn from the rule context's generator, `ctx.rng`, which the engine owns. A
known inverse is not used, and is ignored with a warning.
"""
Base.@kwdef struct CVIProjection{S, OF, IFS, PD, SS} <: MessagePassingRulesApproximations.AbstractApproximationMethod
    outsamples::S = 100
    out_prjparams::OF = nothing
    in_prjparams::IFS = nothing
    proposal_distribution::PD = ProposalDistributionContainer{Any}(nothing)
    sampling_strategy::SS = FullSampling(10)
end

"""
    get_kth_in_form(method::CVIProjection, k)

The projection family `method` names for input `k`, its `in_prjparams.in_k`, or `nothing`.
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
const CVIApproximation = DeltaApproximation{<:CVIProjection}

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

export CVIProjection

export CVISamplingStrategy, FullSampling, MeanBased

@enum CVISamplingStrategy begin
    FullSampling
    MeanBased
end

mutable struct ProposalDistributionContainer{PD}
    distribution::PD
end

"""
    CVIProjection(; parameters...)

A structure representing the parameters for the Conjugate Variational Inference (CVI) projection method. 
This structure is a subtype of `AbstractApproximationMethod` and is used to configure the settings for CVI.

!!! note
    The `CVIProjection` method requires `ExponentialFamilyProjection` package installed in the current environment.

# Parameters

- `rng::R`: The random number generator used for sampling. Default is `Random.MersenneTwister(42)`.
- `marginalsamples::S`: The number of samples used for approximating marginal distributions. Default is `10`.
- `outsamples::S`: The number of samples used for approximating output message distributions. Default is `100`.
- `out_prjparams::OF`: the form parameter used to select the distribution form on which one to project out edge, if it's not provided will be infered from marginal form
- `in_prjparams::IFS`: a namedtuple like object to select the form on which one to project in the input edge, if it's not provided will be infered from the incoming message onto this edge
- `proposal_distribution::PD`: the proposal distribution used for generating samples, if it's not provided will be infered from the incoming message onto this edge
- `sampling_strategy::SS`: the sampling strategy for the logpdf approximation

!!! note
    The `CVIProjection` method is an experimental enhancement of the now-deprecated `CVI`, offering better stability and improved accuracy. 
    Note that the parameters of this structure, as well as their defaults, are subject to change during the experimentation phase.
"""
Base.@kwdef struct CVIProjection{R, S, OF, IFS, PD, SS} <: AbstractApproximationMethod
    rng::R = Random.MersenneTwister(42)
    marginalsamples::S = 10
    outsamples::S = 100
    out_prjparams::OF = nothing
    in_prjparams::IFS = nothing
    proposal_distribution::PD = ProposalDistributionContainer{Any}(nothing)
    sampling_strategy::SS = FullSampling
end

function get_kth_in_form(::CVIProjection{R, S, OF, Nothing}, ::Int) where {R, S, OF}
    return nothing
end

function get_kth_in_form(method::CVIProjection, k::Int)
    key = Symbol("in_$k")
    return get(method.in_prjparams, key, nothing)
end

# This method should only be invoked if a user did not install `ExponentialFamilyProjection`
# in the current Julia session
check_delta_node_compatibility(::Val{false}, ::CVIProjection) = error("CVI projection requires `using ExponentialFamilyProjection` in the current session.")

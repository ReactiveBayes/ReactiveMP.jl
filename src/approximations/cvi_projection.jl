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
- `out_prjparams::OF`: The form parameter used to select the distribution form on which one to project out edge. If not provided, will be inferred from marginal form.
- `in_prjparams::IFS`: A namedtuple-like object to select the form on which one to project in the input edge. If not provided, will be inferred from the incoming message onto this edge.
- `proposal_distribution::PD`: The proposal distribution used for generating samples. If not provided, will be inferred from incoming messages and updated automatically during iterations for improved convergence.
- `sampling_strategy::SS`: The sampling strategy for the logpdf approximation:
  - `FullSampling`: Uses multiple samples drawn from distributions (default). Provides more accurate approximation at the cost of increased computation time.
  - `MeanBased`: Uses only the mean of each distribution as a single sample. Significantly faster but less accurate for non-linear nodes or complex distributions.

# Examples

```julia
# Standard CVI projection with default settings
method = CVIProjection()

# Fast approximation using mean-based sampling
method = CVIProjection(sampling_strategy = MeanBased)

# Custom proposal with increased sample count
proposal = FactorizedJoint((NormalMeanVariance(0.0, 1.0), NormalMeanVariance(0.0, 1.0)))
method = CVIProjection(
    marginalsamples = 100,
    proposal_distribution = ProposalDistributionContainer(proposal)
)
```

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

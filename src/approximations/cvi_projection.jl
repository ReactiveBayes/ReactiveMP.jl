export CVIProjection, ProjectionForm

struct ProjectionForm{T, D, C}
    typeform::T
    dims::D
    conditioner::C
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
- `prjparams::P`: Parameters for the exponential family projection. Default is `nothing`, in which case it will use `ExponentialFamilyProjection.DefaultProjectionParameters()`.
- `target_out_form::OF`: the form parameter used to select the distribution form on which one to project out edge, if it's not provided will be infered from marginal form
- `target_in_forms::IFS`: a namedtuple like object to select the form on which one to project in the input edge, if it's not provided will be infered from the incoming message onto this edge

!!! note
    The `CVIProjection` method is an experimental enhancement of the now-deprecated `CVI`, offering better stability and improved accuracy. 
    Note that the parameters of this structure, as well as their defaults, are subject to change during the experimentation phase.
"""
Base.@kwdef struct CVIProjection{R, S, P, OF, IFS} <: AbstractApproximationMethod
    rng::R = Random.MersenneTwister(42)
    marginalsamples::S = 10
    outsamples::S = 100
    prjparams::P = nothing # ExponentialFamilyProjection.DefaultProjectionParameters()
    target_out_form::OF = nothing
    target_in_forms::IFS = nothing
end

function get_kth_in_form(::CVIProjection{R,S,P,OF,Nothing}, ::Int) where {R,S,P,OF}
    return nothing
end

function get_kth_in_form(method::CVIProjection{R,S,P,OF,NT}, k::Int) where {R,S,P,OF,NT}
    key = Symbol("in_$k")
    return get(method.target_in_forms, key, nothing)
end

# This method should only be invoked if a user did not install `ExponentialFamilyProjection`
# in the current Julia session
check_delta_node_compatibility(::Val{false}, ::CVIProjection) = error("CVI projection requires `using ExponentialFamilyProjection` in the current session.")

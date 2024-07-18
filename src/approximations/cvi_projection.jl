export CVIProjection

Base.@kwdef struct CVIProjection{R, S, P} <: AbstractApproximationMethod 
    rng::R = Random.MersenneTwister(42)
    marginalsamples::S = 10
    outsamples::S = 100
    prjparams::P = nothing # ExponentialFamilyProjection.DefaultProjectionParameters()
end

# This method should only be invoked if a user did not install `ExponentialFamilyProjection`
# in the current Julia session
ReactiveMP.check_compatibility(::Val{false}, ::CVIProjection) = error("CVI projection requires `using ExponentialFamilyProjection` in the current session.")

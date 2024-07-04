module ReactiveMPProjectionExt

using ReactiveMP, ExponentialFamily, ExponentialFamilyProjection, BayesBase

export CVIProjection

struct CVIProjection <: ReactiveMP.AbstractApproximationMethod end

include("layout/cvi_projection.jl")
include("rules/in.jl")
include("rules/out.jl")
include("rules/marginals.jl")

end
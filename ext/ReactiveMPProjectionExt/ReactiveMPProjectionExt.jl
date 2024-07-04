module ReactiveMPProjectionExt

using ReactiveMP, ExponentialFamilyProjection

export CVIProjection

struct CVIProjection <: ReactiveMP.AbstractApproximationMethod end

include("layout/cvi_projection.jl")
include("rules/in.jl")
include("rules/out.jl")
include("rules/marginals.jl")

end
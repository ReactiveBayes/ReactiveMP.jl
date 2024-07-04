module ReactiveMPProjectionExt

export CVIProjection

struct CVIProjection <: AbstractApproximationMethod end

include("layout/cvi_projection.jl")
include("rules/in.jl")
include("rules/out.jl")
include("rules/marginals.jl")

end
"""
    MessagePassingRulesApproximations

Numerical utilities for propagating moments through a deterministic function: the unscented
transform, local linearization, and the Rauch–Tung–Striebel smoother they are paired with. They work on means and
covariances, never on distributions, and know nothing of message passing: a node package's
rules use them.
"""
module MessagePassingRulesApproximations

using LinearAlgebra
using FastCholesky: cholinv, cholsqrt
import ForwardDiff

export AbstractApproximationMethod, approximation_name, approximation_short_name
export Unscented, UT, UnscentedTransform
export approximate, unscented_statistics, sigma_points_weights, smoothRTS
export Linearization, local_linearization

include("approximations.jl")
include("shared.jl")
include("unscented.jl")
include("linearization.jl")
include("rts.jl")

end

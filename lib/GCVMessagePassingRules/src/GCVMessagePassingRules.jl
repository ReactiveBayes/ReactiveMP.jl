"""
    GCVMessagePassingRules

The GCV node, the Gaussian controlled variance `y ~ N(x, exp(κz + ω))`, with its algorithm
[`GCVApproximation`](@ref) and the distribution of its messages towards `z`, `κ` and `ω`,
[`ExponentialLinearQuadratic`](@ref). NormalMeanVariance and NormalMeanPrecision take such a
message on `out` through rules this package adds.
"""
module GCVMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: AbstractAlgorithm
using MessagePassingRulesApproximations: AbstractApproximationMethod, GaussHermiteCubature, approximate_meancov
using StatsFuns: log2π
import StandardMessagePassingRules

export GCV, GCVApproximation, ExponentialLinearQuadratic

include("exponential_linear_quadratic.jl")
include("node.jl")
include("rules.jl")
include("gaussian_extension.jl")

end

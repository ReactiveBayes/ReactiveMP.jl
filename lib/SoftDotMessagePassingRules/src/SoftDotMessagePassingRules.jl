"""
    SoftDotMessagePassingRules

The SoftDot node, `y ~ N(θᵀx, γ⁻¹)`, the dot product of `θ` and `x` softened by Gaussian
noise of precision `γ`, with its variational rules, mean-field or structured `q(y, x)`. It runs
under `DefaultAlgorithm`, and has no belief-propagation rules.
"""
module SoftDotMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using LinearAlgebra: dot
using StatsFuns: log2π
import StandardMessagePassingRules

export SoftDot, softdot

include("helpers.jl")
include("node.jl")
include("rules.jl")

end

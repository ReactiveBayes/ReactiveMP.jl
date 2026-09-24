"""
    ContinuousTransitionMessagePassingRules

The ContinuousTransition node, `y ~ N(f(a) x, W⁻¹)`, a transition from `x` to `y` through a
matrix built from the vector `a`, and its variational rules, mean-field or structured
`q(y, x)`. They run under [`CTVMP`](@ref), which carries `f` and which the model must give.
"""
module ContinuousTransitionMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: AbstractAlgorithm
using ExponentialFamily: WishartFast
using FastCholesky: cholinv
using LinearAlgebra: tr, logdet
import ForwardDiff
import StandardMessagePassingRules

export ContinuousTransition, CTransition, CTVMP

include("node.jl")
include("rules.jl")

end

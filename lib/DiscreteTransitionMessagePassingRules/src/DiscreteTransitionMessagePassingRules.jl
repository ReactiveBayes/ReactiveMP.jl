"""
    DiscreteTransitionMessagePassingRules

[`DiscreteTransition`](@ref), `out ~ Categorical(A[:, in, T1, …, Tn])`, a factor over a tensor of
transition probabilities `A` with any number of conditioning categoricals `T`, as a tensor node:
each of its rules is one tensor contraction, written once for any factorisation.
"""
module DiscreteTransitionMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: rule_inputs, Target
using BayesBase: clamplog, tiny, huge, convert_paramfloattype
using ExponentialFamily: softmax!
using LinearAlgebra: normalize!

export DiscreteTransition

include("tensors.jl")
include("node.jl")
include("rules.jl")

end

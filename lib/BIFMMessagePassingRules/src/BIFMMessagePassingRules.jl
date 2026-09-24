"""
    BIFMMessagePassingRules

[`BIFM`](@ref), a whole time slice of a linear state-space model for backward-information-filter
forward-marginal (BIFM) smoothing, and [`BIFMHelper`](@ref), which turns the backward pass into
the forward one at the start of the chain. The rules run under [`BIFMSmoother`](@ref), which
the model must give.
"""
module BIFMMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: AbstractAlgorithm
using BayesBase: TerminalProdArgument, promote_samplefloattype
using FastCholesky: cholinv
using LinearAlgebra: I, mul!

export BIFM, BIFMHelper, BIFMSmoother

include("bifm.jl")
include("bifm_helper.jl")

end

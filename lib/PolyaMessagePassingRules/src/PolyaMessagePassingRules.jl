"""
    PolyaMessagePassingRules

The Pólya-Gamma augmented nodes: [`BinomialPolya`](@ref), a binomial regression through the
logistic, and [`MultinomialPolya`](@ref), a multinomial through logistic stick-breaking. The
augmentation makes their messages towards the weights normal.

This package is GPL-3 licensed through its dependency PolyaGammaHybridSamplers.
"""
module PolyaMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: AbstractAlgorithm
using MessagePassingRulesApproximations: ghcubature, getweights, getpoints
using BayesBase: promote_samplefloattype, promote_variate_type
using LinearAlgebra: dot, Diagonal
using LogExpFunctions: logistic, softplus
using SpecialFunctions: loggamma
using PolyaGammaHybridSamplers: PolyaGammaHybridSampler

export BinomialPolya, BinomialPolyaApproximation
export MultinomialPolya, MultinomialPolyaApproximation, logistic_stick_breaking, compose_Nks

include("binomial_polya.jl")
include("multinomial_polya.jl")

end

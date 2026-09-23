"""
    StandardMessagePassingRules

The message passing rules for the standard nodes: distributions, arithmetic, logic and the
mixtures, written with `MessagePassingRulesBase`. Every distribution node runs under
`DefaultAlgorithm`, and its rules name no algorithm: whether a rule computes a belief
propagation, variational or structured update follows from the factorisation, through the
engine's default dependency scheme, which gives each rule the messages inside its own cluster
and the marginals of the other clusters, as in v6. A node whose rules ignore the
factorisation declares an algorithm of its own, as [`NormalMixture`](@ref) does.
"""
module StandardMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: annotate!
using StatsFuns: log2π
using SpecialFunctions: loggamma, logfactorial, logbeta, digamma, gamma
using Base.Broadcast: BroadcastFunction
using BayesBase: tiny, mirrorlog, LinearizedProductOf, MixtureDistribution, TerminalProdArgument
using LogExpFunctions: softmax!
using BayesBase: ClosedProd, PreserveTypeProd, ContinuousUnivariateLogPdf
import DomainSets

export NormalMixture, NormalMixtureVMP, GammaShapeLikelihood, HalfNormal, Uninformative

include("helpers.jl")

# The nodes this package declares, for types other packages own.
const NODES = [NormalMeanVariance, NormalMeanPrecision, GammaShapeRate, Categorical, Dirichlet, Beta, Bernoulli, Gamma, GammaInverse, Poisson, Uniform]

include("nodes/normal_mean_variance.jl")
include("rules/normal_mean_variance/out.jl")
include("rules/normal_mean_variance/mean.jl")
include("rules/normal_mean_variance/var.jl")
include("rules/normal_mean_variance/marginals.jl")

include("nodes/normal_mean_precision.jl")
include("rules/normal_mean_precision/out.jl")
include("rules/normal_mean_precision/mean.jl")
include("rules/normal_mean_precision/precision.jl")
include("rules/normal_mean_precision/marginals.jl")

include("nodes/gamma_shape_rate.jl")
include("gamma_shape_likelihood.jl")
include("rules/gamma_shape_rate/out.jl")
include("rules/gamma_shape_rate/a.jl")
include("rules/gamma_shape_rate/b.jl")
include("rules/gamma_shape_rate/marginals.jl")

include("nodes/categorical.jl")
include("rules/categorical/out.jl")
include("rules/categorical/p.jl")
include("rules/categorical/marginals.jl")

include("nodes/dirichlet.jl")
include("rules/dirichlet/out.jl")
include("rules/dirichlet/marginals.jl")

include("nodes/beta.jl")
include("rules/beta/out.jl")
include("rules/beta/marginals.jl")

include("nodes/bernoulli.jl")
include("rules/bernoulli/out.jl")
include("rules/bernoulli/p.jl")
include("rules/bernoulli/marginals.jl")

include("nodes/gamma.jl")
include("rules/gamma/out.jl")
include("rules/gamma/marginals.jl")

include("nodes/gamma_inverse.jl")
include("rules/gamma_inverse/out.jl")
include("rules/gamma_inverse/marginals.jl")

include("nodes/half_normal.jl")
include("rules/half_normal/out.jl")

include("nodes/poisson.jl")
include("rules/poisson/out.jl")
include("rules/poisson/l.jl")
include("rules/poisson/marginals.jl")

include("nodes/uniform.jl")
include("rules/uniform/out.jl")

include("nodes/uninformative.jl")
include("rules/uninformative/out.jl")

include("nodes/normal_mixture.jl")
include("rules/normal_mixture/m.jl")
include("rules/normal_mixture/p.jl")
include("rules/normal_mixture/switch.jl")
include("rules/normal_mixture/out.jl")

end

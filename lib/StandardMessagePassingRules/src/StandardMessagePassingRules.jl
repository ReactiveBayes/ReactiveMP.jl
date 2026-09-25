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
using MessagePassingRulesBase: annotate!, matrix_correction, hasannotation, getannotation
using StatsFuns: log2π, logπ
using SpecialFunctions: loggamma, logfactorial, logbeta, digamma, gamma, besselk
using Base.Broadcast: BroadcastFunction
using BayesBase: tiny, mirrorlog, LinearizedProductOf, MixtureDistribution, TerminalProdArgument
using LogExpFunctions: softmax!, softmax, logsumexp
using BayesBase: ClosedProd, PreserveTypeProd, ContinuousUnivariateLogPdf
import LinearAlgebra
using LinearAlgebra: I, Hermitian, UniformScaling, tr, logdet, dot
using FastCholesky: cholinv, fastcholesky
using MatrixCorrectionTools: correction!, ReplaceZeroDiagonalEntries
import ExponentialFamily: InverseWishartFast, WishartFast, WishartDistributionsFamily, InverseWishartDistributionsFamily, covmats
import DomainSets

export NormalMixture, GaussianMixture, NormalMixtureVMP, GammaMixture, GammaMixtureVMP, Mixture, MixtureBP, GammaShapeLikelihood, HalfNormal, Uninformative
export AND, OR, NOT, IMPLY, MultiplicationSampling

include("helpers.jl")

# The nodes this package declares, for types other packages own.
const NODES = [
    NormalMeanVariance, NormalMeanPrecision, GammaShapeRate, Categorical, Dirichlet, Beta, Bernoulli, Gamma, GammaInverse, Poisson, Uniform,
    MvNormalMeanCovariance, MvNormalMeanPrecision, MvNormalWeightedMeanPrecision, MvNormalMeanScalePrecision,
    MvNormalMeanScaleMatrixPrecision, Wishart, InverseWishart, DirichletCollection, MvNormalGamma,
    MvNormalWishart, MatrixNormal, MatrixNormalWishart, +, -, dot, *,
]

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

include("nodes/mv_normal_mean_covariance.jl")
include("rules/mv_normal_mean_covariance/out.jl")
include("rules/mv_normal_mean_covariance/mean.jl")
include("rules/mv_normal_mean_covariance/covariance.jl")
include("rules/mv_normal_mean_covariance/marginals.jl")

include("nodes/mv_normal_mean_precision.jl")
include("rules/mv_normal_mean_precision/out.jl")
include("rules/mv_normal_mean_precision/mean.jl")
include("rules/mv_normal_mean_precision/precision.jl")
include("rules/mv_normal_mean_precision/marginals.jl")

include("nodes/mv_normal_weighted_mean_precision.jl")
include("rules/mv_normal_weighted_mean_precision/out.jl")
include("rules/mv_normal_weighted_mean_precision/marginals.jl")

include("nodes/mv_normal_mean_scale_precision.jl")
include("rules/mv_normal_mean_scale_precision/out.jl")
include("rules/mv_normal_mean_scale_precision/mean.jl")
include("rules/mv_normal_mean_scale_precision/precision.jl")
include("rules/mv_normal_mean_scale_precision/marginals.jl")
include("nodes/mv_normal_mean_scale_matrix_precision.jl")
include("rules/mv_normal_mean_scale_matrix_precision/out.jl")
include("rules/mv_normal_mean_scale_matrix_precision/mean.jl")
include("rules/mv_normal_mean_scale_matrix_precision/precision.jl")
include("rules/mv_normal_mean_scale_matrix_precision/matrix.jl")
include("rules/mv_normal_mean_scale_matrix_precision/marginals.jl")
include("nodes/wishart.jl")
include("rules/wishart/out.jl")
include("rules/wishart/marginals.jl")
include("nodes/inverse_wishart.jl")
include("rules/inverse_wishart/out.jl")
include("rules/inverse_wishart/marginals.jl")
include("nodes/dirichlet_collection.jl")
include("rules/dirichlet_collection/out.jl")
include("rules/dirichlet_collection/marginals.jl")
include("nodes/mv_normal_gamma.jl")
include("rules/mv_normal_gamma/out.jl")
include("nodes/mv_normal_wishart.jl")
include("rules/mv_normal_wishart/out.jl")
include("nodes/matrix_normal.jl")
include("rules/matrix_normal/out.jl")
include("rules/matrix_normal/M.jl")
include("rules/matrix_normal/U.jl")
include("rules/matrix_normal/V.jl")
include("rules/matrix_normal/marginals.jl")
include("nodes/matrix_normal_wishart.jl")
include("rules/matrix_normal_wishart/out.jl")
include("nodes/addition.jl")
include("rules/addition/out.jl")
include("rules/addition/in1.jl")
include("rules/addition/in2.jl")
include("rules/addition/marginals.jl")
include("nodes/subtraction.jl")
include("rules/subtraction/out.jl")
include("rules/subtraction/in1.jl")
include("rules/subtraction/in2.jl")
include("rules/subtraction/marginals.jl")
include("nodes/dot_product.jl")
include("rules/dot_product/out.jl")
include("rules/dot_product/in1.jl")
include("rules/dot_product/in2.jl")
include("rules/dot_product/marginals.jl")
include("nodes/multiplication.jl")
include("rules/multiplication/out.jl")
include("rules/multiplication/in.jl")
include("rules/multiplication/A.jl")
include("rules/multiplication/marginals.jl")

include("nodes/logic.jl")
include("rules/and/rules.jl")
include("rules/or/rules.jl")
include("rules/not/rules.jl")
include("rules/implication/rules.jl")

include("nodes/normal_mixture.jl")
include("rules/normal_mixture/m.jl")
include("rules/normal_mixture/p.jl")
include("rules/normal_mixture/switch.jl")
include("rules/normal_mixture/out.jl")
include("nodes/gamma_mixture.jl")
include("rules/gamma_mixture/a.jl")
include("rules/gamma_mixture/b.jl")
include("rules/gamma_mixture/out.jl")
include("rules/gamma_mixture/switch.jl")
include("nodes/mixture.jl")
include("rules/mixture/inputs.jl")
include("rules/mixture/out.jl")
include("rules/mixture/switch.jl")

end

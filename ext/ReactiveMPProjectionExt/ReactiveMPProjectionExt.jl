module ReactiveMPProjectionExt

using ReactiveMP, ExponentialFamily, AdvancedHMC, LogDensityProblems, Distributions, ExponentialFamilyProjection, BayesBase, Random, LinearAlgebra, FastCholesky
using ForwardDiff
export CVIProjection, DivisionOf

Base.@kwdef struct CVIProjection{OS, MS, P, R} <: ReactiveMP.AbstractApproximationMethod 
    out_samples_no::OS = 100
    marginal_samples_no::MS = 10
    projection_parameters::P = ExponentialFamilyProjection.DefaultProjectionParameters()
    rng::R = Random.MersenneTwister(42)
end

struct DivisionOf{A, B}
    numerator::A
    denumerator::B
end

BayesBase.insupport(d::DivisionOf, p) = insupport(d.numerator, p) && insupport(d.denumerator, p)
BayesBase.logpdf(d::DivisionOf, p) = logpdf(d.numerator, p) - logpdf(d.denumerator, p)


# cost function
function targetfn(M, p, data)
    ef = convert(ExponentialFamilyDistribution, M, p)
    return -mean(logpdf(ef, data))
end

# # gradient function
## I think this is wrong. This is not a gradient on the manifolds. It is just Euclidean gradient.
function grad_targetfn(M, p, data)
    ef = convert(ExponentialFamilyDistribution, M, p)
    fisher = cholinv(Hermitian(fisherinformation(ef)))
    X = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(M, fisher*ForwardDiff.gradient((p) -> targetfn(M, p, data),p))
    return ExponentialFamilyProjection.Manopt.project(M, p, X)
end

struct LogTargetDensity{F}
    dim::Int
    f :: F
end

LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
LogDensityProblems.capabilities(::LogTargetDensity) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.logdensity(p::LogTargetDensity, x) = p.f(x)  


include("layout/cvi_projection.jl")
include("rules/in.jl")
include("rules/out.jl")
include("rules/marginals.jl")

end
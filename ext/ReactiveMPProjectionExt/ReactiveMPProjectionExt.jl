module ReactiveMPProjectionExt

using ReactiveMP, ExponentialFamily, Distributions, ExponentialFamilyProjection, BayesBase, Random, LinearAlgebra, FastCholesky

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

include("layout/cvi_projection.jl")
include("rules/in.jl")
include("rules/out.jl")
include("rules/marginals.jl")

end
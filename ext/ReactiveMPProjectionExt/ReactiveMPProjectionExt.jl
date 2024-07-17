module ReactiveMPProjectionExt

using ReactiveMP, ExponentialFamily, Distributions, ExponentialFamilyProjection, BayesBase, Random, LinearAlgebra, FastCholesky

export CVIProjection

Base.@kwdef struct CVIProjection{R, S, P} <: ReactiveMP.AbstractApproximationMethod 
    rng::R = Random.MersenneTwister(42)
    marginalsamples::S = 100
    outsamples::S = 500
    prjparams::P = ExponentialFamilyProjection.DefaultProjectionParameters()
end

struct DivisionOf{A, B}
    numerator::A
    denumerator::B
end

BayesBase.insupport(d::DivisionOf, p) = insupport(d.numerator, p) && insupport(d.denumerator, p)
BayesBase.logpdf(d::DivisionOf, p) = logpdf(d.numerator, p) - logpdf(d.denumerator, p)

function BayesBase.prod(::GenericProd, something, division::DivisionOf) 
    return prod(GenericProd(), division, something)
end

function BayesBase.prod(::GenericProd, division::DivisionOf, something)
    if division.denumerator == something
        return division.numerator
    else
        return ProductOf(division, something)
    end
end

include("layout/cvi_projection.jl")
include("rules/in.jl")
include("rules/out.jl")
include("rules/marginals.jl")

end
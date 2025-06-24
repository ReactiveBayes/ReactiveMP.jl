module ReactiveMPProjectionExt

using ReactiveMP, ExponentialFamily, Distributions, ExponentialFamilyProjection, BayesBase, Random, LinearAlgebra, FastCholesky

struct DivisionOf{A, B}
    numerator::A
    denumerator::B
end

(divisionof::DivisionOf)(x) = logpdf(divisionof, x)
BayesBase.insupport(d::DivisionOf, p) = insupport(d.numerator, p) && insupport(d.denumerator, p)
BayesBase.logpdf(d::DivisionOf, p) = logpdf(d.numerator, p) - logpdf(d.denumerator, p)

function BayesBase.prod(::GenericProd, something::DivisionOf, division::DivisionOf)
    if division.denumerator == something.numerator
        return DivisionOf(division.numerator, something.denumerator)
    elseif division.numerator == something.denumerator
        return DivisionOf(something.numerator, division.denumerator)
    else
        return ProductOf(something, division)
    end
end

function BayesBase.prod(::GenericProd, something, division::DivisionOf)
    return prod(GenericProd(), division, something)
end

function BayesBase.prod(::GenericProd, division::DivisionOf, something::Any)
    if division.denumerator == something
        return division.numerator
    else
        return ProductOf(division, something)
    end
end

BayesBase.prod(::GenericProd, division::DivisionOf, ::Missing) = division
BayesBase.prod(::GenericProd, ::Missing, division::DivisionOf) = division

function BayesBase.prod(::GenericProd, productof::ProductOf, divisionof::DivisionOf)
    return ProductOf(productof, divisionof)
end

include("layout/cvi_projection.jl")
include("rules/in.jl")
include("rules/out.jl")
include("rules/marginals.jl")
include("divisionof/univariate_gaussian.jl")
include("divisionof/multivariate_gaussian.jl")

# This will enable the extension and make `CVIProjection` compatible with delta nodes 
# Otherwise it should throw an error suggesting users to install `ExponentialFamilyProjection`
# See `approximations/cvi_projection.jl`
ReactiveMP.is_delta_node_compatible(::ReactiveMP.CVIProjection) = Val(true)

end

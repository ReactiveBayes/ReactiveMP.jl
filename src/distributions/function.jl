export ContinuousUnivariateLogPdf

using Distributions

import DomainSets
import DomainIntegrals
import Base: isapprox

"""
    ContinuousUnivariateLogPdf{ D <: DomainSets.Domain, F } <: ContinuousUnivariateDistribution

Generic continuous univariate distribution in a form of domain specification and logpdf function. Can be used in cases where no 
known analytical distribution available. 

# Arguments 
- `domain`: domain specificatiom from `DomainSets.jl` package
- `logpdf`: callable object that represents a `logpdf` of a distribution. Does not necessarily normalised.

```julia 
fdist = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> -x^2)
```
"""
struct ContinuousUnivariateLogPdf{D <: DomainSets.Domain, F} <: ContinuousUnivariateDistribution
    domain::D
    logpdf::F
end

ContinuousUnivariateLogPdf(f::Function) = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), f)

(dist::ContinuousUnivariateLogPdf)(x::Real)                   = logpdf(dist, x)
(dist::ContinuousUnivariateLogPdf)(x::AbstractVector{<:Real}) = logpdf(dist, x)

Distributions.support(dist::ContinuousUnivariateLogPdf) =
    Distributions.RealInterval(DomainSets.infimum(dist.domain), DomainSets.supremum(dist.domain))

Distributions.mean(dist::ContinuousUnivariateLogPdf)    = error("mean() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.median(dist::ContinuousUnivariateLogPdf)  = error("median() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.mode(dist::ContinuousUnivariateLogPdf)    = error("mode() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.var(dist::ContinuousUnivariateLogPdf)     = error("var() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.std(dist::ContinuousUnivariateLogPdf)     = error("std() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.cov(dist::ContinuousUnivariateLogPdf)     = error("cov() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.invcov(dist::ContinuousUnivariateLogPdf)  = error("invcov() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.entropy(dist::ContinuousUnivariateLogPdf) = error("entropy() is not defined for `ContinuousUnivariateLogPdf`.")

# We don't expect neither `pdf` nor `logpdf` to be normalised
Distributions.pdf(dist::ContinuousUnivariateLogPdf, x::Real) = exp(logpdf(dist, x))

function Distributions.logpdf(dist::ContinuousUnivariateLogPdf, x::Real)
    @assert x ∈ dist.domain "x = $(x) does not belong to the domain of $dist"
    return dist.logpdf(x)
end

# These are fallbacks for various optimisation packages which may pass arguments as vectors
function Distributions.pdf(dist::ContinuousUnivariateLogPdf, x::AbstractVector{<:Real})
    @assert length(x) === 1 "`ContinuousUnivariateLogPdf` expects either float or a vector of a single float as an input for the `pdf` function."
    return exp(logpdf(dist, first(x)))
end

function Distributions.logpdf(dist::ContinuousUnivariateLogPdf, x::AbstractVector{<:Real})
    @assert length(x) === 1 "`ContinuousUnivariateLogPdf` expects either float or a vector of a single float as an input for the `logpdf` function."
    return logpdf(dist, first(x))
end

Base.precision(dist::ContinuousUnivariateLogPdf) = error("precision() is not defined for `ContinuousUnivariateLogPdf`.")

Base.convert(::Type{ContinuousUnivariateLogPdf}, domain::D, logpdf::F) where {D <: DomainSets.Domain, F} =
    ContinuousUnivariateLogPdf{D, F}(domain, logpdf)

convert_eltype(::Type{ContinuousUnivariateLogPdf}, ::Type{T}, dist::ContinuousUnivariateLogPdf) where {T <: Real} =
    convert(ContinuousUnivariateLogPdf, dist.domain, dist.logpdf)

vague(::Type{<:ContinuousUnivariateLogPdf}) = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> 1.0)

prod_analytical_rule(::Type{<:ContinuousUnivariateLogPdf}, ::Type{<:ContinuousUnivariateLogPdf}) =
    ProdAnalyticalRuleAvailable()

function prod(
    ::ProdAnalytical,
    left::ContinuousUnivariateLogPdf{D1, F1},
    right::ContinuousUnivariateLogPdf{D2, F2}
) where {D1, D2, F1, F2}
    @assert left.domain == right.domain "Different domain types in product of generic `ContinuousUnivariateLogPdf` distributions. Left domain is $(left.domain), right is $(right.domain)."
    plogpdf = let left = left, right = right
        (x) -> logpdf(left, x) + logpdf(right, x)
    end
    return ContinuousUnivariateLogPdf(left.domain, plogpdf)
end

## More efficient prod for same logpdfs

struct ContinuousUnivariateLogPdfVectorisedProduct{F} <: ContinuousUnivariateDistribution
    vector::Vector{F}
end

(dist::ContinuousUnivariateLogPdfVectorisedProduct)(x) = logpdf(dist, x)

Distributions.support(dist::ContinuousUnivariateLogPdfVectorisedProduct) = Distributions.support(first(dist.vector))

Distributions.mean(dist::ContinuousUnivariateLogPdfVectorisedProduct)    = error("mean() is not defined for `ContinuousUnivariateLogPdfVectorisedProduct`.")
Distributions.median(dist::ContinuousUnivariateLogPdfVectorisedProduct)  = error("median() is not defined for `ContinuousUnivariateLogPdfVectorisedProduct`.")
Distributions.mode(dist::ContinuousUnivariateLogPdfVectorisedProduct)    = error("mode() is not defined for `ContinuousUnivariateLogPdfVectorisedProduct`.")
Distributions.var(dist::ContinuousUnivariateLogPdfVectorisedProduct)     = error("var() is not defined for `ContinuousUnivariateLogPdfVectorisedProduct`.")
Distributions.std(dist::ContinuousUnivariateLogPdfVectorisedProduct)     = error("std() is not defined for `ContinuousUnivariateLogPdfVectorisedProduct`.")
Distributions.cov(dist::ContinuousUnivariateLogPdfVectorisedProduct)     = error("cov() is not defined for `ContinuousUnivariateLogPdfVectorisedProduct`.")
Distributions.invcov(dist::ContinuousUnivariateLogPdfVectorisedProduct)  = error("invcov() is not defined for `ContinuousUnivariateLogPdfVectorisedProduct`.")
Distributions.entropy(dist::ContinuousUnivariateLogPdfVectorisedProduct) = error("entropy() is not defined for `ContinuousUnivariateLogPdfVectorisedProduct`.")

Distributions.logpdf(dist::ContinuousUnivariateLogPdfVectorisedProduct, x) = mapreduce((d) -> logpdf(d, x), +, dist.vector)
Distributions.pdf(dist::ContinuousUnivariateLogPdfVectorisedProduct, x)    = exp(mapreduce((d) -> logpdf(d, x), +, dist.vector))

function prod(
    ::ProdAnalytical,
    left::ContinuousUnivariateLogPdf{D, F},
    right::ContinuousUnivariateLogPdf{D, F}
) where {D, F}
    return ContinuousUnivariateLogPdfVectorisedProduct(ContinuousUnivariateLogPdf{D, F}[left, right])
end

prod_analytical_rule(
    ::Type{ContinuousUnivariateLogPdfVectorisedProduct{F}},
    ::Type{F}
) where {F <: ContinuousUnivariateLogPdf} = ProdAnalyticalRuleAvailable()

function prod(
    ::ProdAnalytical,
    left::ContinuousUnivariateLogPdfVectorisedProduct{F},
    right::F
) where {F <: ContinuousUnivariateLogPdf}
    push!(left.vector, right)
    return left
end

# This method is inaccurate and relies on various approximation methods, which may fail in different scenarious
# Current implementation of `isapprox` method supports only FullSpace and HalfLine domains with limited accuracy
function Base.isapprox(left::ContinuousUnivariateLogPdf, right::ContinuousUnivariateLogPdf; kwargs...)
    if left.domain !== right.domain
        return false
    end
    return culogpdf__isapprox(left.domain, left, right; kwargs...)
end

# https://en.wikipedia.org/wiki/Gauss–Hermite_quadrature
function culogpdf__isapprox(
    domain::DomainSets.FullSpace,
    left::ContinuousUnivariateLogPdf,
    right::ContinuousUnivariateLogPdf;
    kwargs...
)
    return isapprox(
        zero(eltype(domain)),
        DomainIntegrals.integral(Q_GaussHermite(32), (x) -> exp(x^2) * abs(left(x) - right(x)));
        kwargs...
    )
end

# https://en.wikipedia.org/wiki/Gauss–Laguerre_quadrature
function culogpdf__isapprox(
    domain::DomainSets.HalfLine,
    left::ContinuousUnivariateLogPdf,
    right::ContinuousUnivariateLogPdf;
    kwargs...
)
    return isapprox(
        zero(eltype(domain)),
        DomainIntegrals.integral(Q_GaussLaguerre(32), (x) -> exp(x) * abs(left(x) - right(x)));
        kwargs...
    )
end

# We do not check typeof of a different functions because in most of the cases lambdas have different types, but it does not really mean that objects are different
function is_typeof_equal(
    left::ContinuousUnivariateLogPdf{D, F1},
    right::ContinuousUnivariateLogPdf{D, F2}
) where {D, F1 <: Function, F2 <: Function}
    return true
end

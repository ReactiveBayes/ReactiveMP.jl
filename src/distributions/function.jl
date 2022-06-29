export ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf

using Distributions

import DomainSets
import DomainIntegrals
import HCubature

import Base: isapprox

abstract type AbstractContinuousGenericLogPdf end

value_support(::Type{<:AbstractContinuousGenericLogPdf}) = Continuous
value_support(::AbstractContinuousGenericLogPdf)         = Continuous

# We throw an error on purpose, since we do not want to use `AbstractContinuousGenericLogPdf` much without approximations
# We want to encourage a user to use functional form constraints and approximate generic log-pdfs as much as possible instead
__error_not_defined(dist::AbstractContinuousGenericLogPdf, f::Symbol) = error(
    "`$f` is not defined for `$(dist)`. Use functional form constraints to approximate the resulting generic log-pdf object and to use it in the inference procedure."
)

Distributions.mean(dist::AbstractContinuousGenericLogPdf)    = __error_not_defined(dist, :mean)
Distributions.median(dist::AbstractContinuousGenericLogPdf)  = __error_not_defined(dist, :median)
Distributions.mode(dist::AbstractContinuousGenericLogPdf)    = __error_not_defined(dist, :mode)
Distributions.var(dist::AbstractContinuousGenericLogPdf)     = __error_not_defined(dist, :var)
Distributions.std(dist::AbstractContinuousGenericLogPdf)     = __error_not_defined(dist, :std)
Distributions.cov(dist::AbstractContinuousGenericLogPdf)     = __error_not_defined(dist, :cov)
Distributions.invcov(dist::AbstractContinuousGenericLogPdf)  = __error_not_defined(dist, :invcov)
Distributions.entropy(dist::AbstractContinuousGenericLogPdf) = __error_not_defined(dist, :entropy)

Base.precision(dist::AbstractContinuousGenericLogPdf) = __error_not_defined(dist, :precision)

Base.eltype(dist::AbstractContinuousGenericLogPdf) = eltype(getdomain(dist))

(dist::AbstractContinuousGenericLogPdf)(x::Real)                   = logpdf(dist, x)
(dist::AbstractContinuousGenericLogPdf)(x::AbstractVector{<:Real}) = logpdf(dist, x)

function Distributions.logpdf(dist::AbstractContinuousGenericLogPdf, x)
    @assert x ∈ getdomain(dist) "x = $(x) does not belong to the domain ($(getdomain(dist))) of $dist"
    lpdf = getlogpdf(dist)
    return lpdf(x)
end

# We don't expect neither `pdf` nor `logpdf` to be normalised
Distributions.pdf(dist::AbstractContinuousGenericLogPdf, x) = exp(logpdf(dist, x))

prod_analytical_rule(::Type{<:AbstractContinuousGenericLogPdf}, ::Type{<:AbstractContinuousGenericLogPdf}) =
    ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::AbstractContinuousGenericLogPdf, right::AbstractContinuousGenericLogPdf)
    @assert value_support(typeof(left)) === value_support(typeof(right)) "Cannot compute a product of $(left) and $(right). Inputs have different value support: $(value_support(typeof(left))) and $(value_support(typeof(right)))"
    @assert variate_form(typeof(left)) === variate_form(typeof(right)) "Cannot compute a product of $(left) and $(right). Inputs have different variate forms: $(variate_form(typeof(left))) and $(variate_form(typeof(right)))"
    @assert getdomain(left) == getdomain(right) "Cannot compute a product of $(left) and $(right). Inputs have different domains: $(getdomain(left)) and $(getdomain(right))."
    plogpdf = let left = left, right = right
        (x) -> logpdf(left, x) + logpdf(right, x)
    end
    return convert(typeof(left), getdomain(left), plogpdf)
end

"""
    ContinuousUnivariateLogPdf{ D <: DomainSets.Domain, F } <: AbstractContinuousGenericLogPdf

Generic continuous univariate distribution in a form of domain specification and logpdf function. Can be used in cases where no 
known analytical distribution available. 

# Arguments 
- `domain`: domain specificatiom from `DomainSets.jl` package
- `logpdf`: callable object that represents a `logpdf` of a distribution. Does not necessarily normalised.

```julia 
fdist = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> -x^2)
```
"""
struct ContinuousUnivariateLogPdf{D <: DomainSets.Domain, F} <: AbstractContinuousGenericLogPdf
    domain::D
    logpdf::F

    ContinuousUnivariateLogPdf(domain::D, logpdf::F) where {D, F} = begin
        @assert DomainSets.dimension(domain) === 1 "Cannot create ContinuousUnivariateLogPdf. Dimension of domain = $(domain) is not equal to 1."
        return new{D, F}(domain, logpdf)
    end
end

variate_form(::Type{<:ContinuousUnivariateLogPdf}) = Univariate
variate_form(::ContinuousUnivariateLogPdf)         = Univariate

getdomain(dist::ContinuousUnivariateLogPdf) = dist.domain
getlogpdf(dist::ContinuousUnivariateLogPdf) = dist.logpdf

ContinuousUnivariateLogPdf(f::Function) = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), f)

Base.show(io::IO, dist::ContinuousUnivariateLogPdf) = print(io, "ContinuousUnivariateLogPdf(", getdomain(dist), ")")
Base.show(io::IO, ::Type{<:ContinuousUnivariateLogPdf{D}}) where {D} = print(io, "ContinuousUnivariateLogPdf{", D, "}")

Distributions.support(dist::ContinuousUnivariateLogPdf) =
    Distributions.RealInterval(DomainSets.infimum(getdomain(dist)), DomainSets.supremum(getdomain(dist)))

# Fallback for various optimisation packages which may pass arguments as vectors
function Distributions.logpdf(dist::ContinuousUnivariateLogPdf, x::AbstractVector{<:Real})
    @assert length(x) === 1 "`ContinuousUnivariateLogPdf` expects either float or a vector of a single float as an input for the `logpdf` function."
    return logpdf(dist, first(x))
end

Base.convert(::Type{<:ContinuousUnivariateLogPdf}, domain::D, logpdf::F) where {D <: DomainSets.Domain, F} =
    ContinuousUnivariateLogPdf(domain, logpdf)

convert_eltype(::Type{ContinuousUnivariateLogPdf}, ::Type{T}, dist::ContinuousUnivariateLogPdf) where {T <: Real} =
    convert(ContinuousUnivariateLogPdf, dist.domain, dist.logpdf)

vague(::Type{<:ContinuousUnivariateLogPdf}) = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> 1.0)

# We do not check typeof of a different functions because in most of the cases lambdas have different types, but they can still be the same
function is_typeof_equal(
    ::ContinuousUnivariateLogPdf{D, F1},
    ::ContinuousUnivariateLogPdf{D, F2}
) where {D, F1 <: Function, F2 <: Function}
    return true
end

## 

"""
    ContinuousMultivariateLogPdf{ D <: DomainSets.Domain, F } <: AbstractContinuousGenericLogPdf

Generic continuous multivariate distribution in a form of domain specification and logpdf function. Can be used in cases where no 
known analytical distribution available. 

# Arguments 
- `domain`: multidimensional domain specificatiom from `DomainSets.jl` package
- `logpdf`: callable object that accepts an `AbstractVector` as an input and represents a `logpdf` of a distribution. Does not necessarily normalised.

```julia 
fdist = ContinuousMultivariateLogPdf(DomainSets.FullSpace() ^ 2, (x) -> -x'x)
```
"""
struct ContinuousMultivariateLogPdf{D <: DomainSets.Domain, F} <: AbstractContinuousGenericLogPdf
    domain::D
    logpdf::F

    ContinuousMultivariateLogPdf(domain::D, logpdf::F) where {D <: DomainSets.Domain, F} = begin
        @assert DomainSets.dimension(domain) !== 1 "Cannot create ContinuousMultivariateLogPdf. Dimension of domain = $(domain) should not be equal to 1. Use, for example, `DomainSets.FullSpace() ^ 2` to create 2-dimensional full space domain."
        return new{D, F}(domain, logpdf)
    end
end

variate_form(::Type{<:ContinuousMultivariateLogPdf}) = Multivariate
variate_form(::ContinuousMultivariateLogPdf)         = Multivariate

getdomain(dist::ContinuousMultivariateLogPdf) = dist.domain
getlogpdf(dist::ContinuousMultivariateLogPdf) = dist.logpdf

ContinuousMultivariateLogPdf(dims::Int, f::Function) = ContinuousMultivariateLogPdf(DomainSets.FullSpace()^dims, f)

Base.show(io::IO, dist::ContinuousMultivariateLogPdf) = print(io, "ContinuousMultivariateLogPdf(", getdomain(dist), ")")
Base.show(io::IO, ::Type{<:ContinuousMultivariateLogPdf{D}}) where {D} =
    print(io, "ContinuousMultivariateLogPdf{", D, "}")

Distributions.support(dist::ContinuousMultivariateLogPdf) = getdomain(dist) # differs from Univariate implementation, todo if it causes any problems

Base.convert(::Type{<:ContinuousMultivariateLogPdf}, domain::D, logpdf::F) where {D <: DomainSets.Domain, F} =
    ContinuousMultivariateLogPdf(domain, logpdf)

convert_eltype(::Type{ContinuousMultivariateLogPdf}, ::Type{T}, dist::ContinuousMultivariateLogPdf) where {T <: Real} =
    convert(ContinuousMultivariateLogPdf, dist.domain, dist.logpdf)

vague(::Type{<:ContinuousMultivariateLogPdf}, dims::Int) =
    ContinuousMultivariateLogPdf(DomainSets.FullSpace()^dims, (x) -> Float64(dims))

# We do not check typeof of a different functions because in most of the cases lambdas have different types, but they can still be the same
function is_typeof_equal(
    ::ContinuousMultivariateLogPdf{D, F1},
    ::ContinuousMultivariateLogPdf{D, F2}
) where {D, F1 <: Function, F2 <: Function}
    return true
end

## More efficient prod for same logpdfs
## We need to implement this auxilary functions, because we define `prod_analytical_rule = ProdAnalyticalRuleAvailable()` for AbstractContinuousGenericLogPdf
## By default `GenericLogPdfVectorisedProduct` works only if `prod_analytical_rule = ProdAnalyticalRuleUnknown()`

prod(::ProdAnalytical, left::F, right::F) where {F <: AbstractContinuousGenericLogPdf} = GenericLogPdfVectorisedProduct(F[left, right], 2)
prod(::ProdAnalytical, left::GenericLogPdfVectorisedProduct{F}, right::F) where {F <: AbstractContinuousGenericLogPdf} = push!(left, right)

## Utility methods for tests 

# These methods are inaccurate and relies on various approximation methods, which may fail in different scenarios
# This should not be used though anywhere in the real code, but only in tests
# Current implementation of `isapprox` method supports only FullSpace and HalfLine domains with limited accuracy
function Base.isapprox(left::AbstractContinuousGenericLogPdf, right::AbstractContinuousGenericLogPdf; kwargs...)
    if (getdomain(left) !== getdomain(right)) || (value_support(typeof(left)) !== value_support(typeof(right))) ||
       (variate_form(typeof(left)) !== variate_form(typeof(right)))
        return false
    end
    return culogpdf__isapprox(getdomain(left), left, right; kwargs...)
end

# https://en.wikipedia.org/wiki/Gauss–Hermite_quadrature
function culogpdf__isapprox(
    domain::DomainSets.FullSpace,
    left::AbstractContinuousGenericLogPdf,
    right::AbstractContinuousGenericLogPdf;
    kwargs...
)
    return isapprox(
        zero(eltype(domain)),
        DomainIntegrals.integral(DomainIntegrals.Q_GaussHermite(32), (x) -> exp(x^2) * abs(left(x) - right(x)));
        kwargs...
    )
end

# https://en.wikipedia.org/wiki/Gauss–Laguerre_quadrature
function culogpdf__isapprox(
    domain::DomainSets.HalfLine,
    left::AbstractContinuousGenericLogPdf,
    right::AbstractContinuousGenericLogPdf;
    kwargs...
)
    return isapprox(
        zero(eltype(domain)),
        DomainIntegrals.integral(DomainIntegrals.Q_GaussLaguerre(32), (x) -> exp(x) * abs(left(x) - right(x)));
        kwargs...
    )
end

function culogpdf__isapprox(
    domain::DomainSets.VcatDomain,
    left::AbstractContinuousGenericLogPdf,
    right::AbstractContinuousGenericLogPdf;
    kwargs...
)
    a = clamp.(DomainSets.infimum(domain), -1e5, 1e5)
    b = clamp.(DomainSets.supremum(domain), -1e5, 1e5)
    (I, E) = HCubature.hcubature((x) -> abs(left(x) - right(x)), a, b)
    return isapprox(zero(deep_eltype(domain)), I; kwargs...) && isapprox(zero(deep_eltype(domain)), E; kwargs...)
end

function culogpdf__isapprox(
    domain::DomainSets.FixedIntervalProduct,
    left::AbstractContinuousGenericLogPdf,
    right::AbstractContinuousGenericLogPdf;
    kwargs...
)
    a = clamp.(DomainSets.infimum(domain), -1e5, 1e5)
    b = clamp.(DomainSets.supremum(domain), -1e5, 1e5)
    (I, E) = HCubature.hcubature((x) -> abs(left(x) - right(x)), a, b)
    return isapprox(zero(deep_eltype(domain)), I; kwargs...) && isapprox(zero(deep_eltype(domain)), E; kwargs...)
end

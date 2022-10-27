export ProdFinal

import Base: eltype

"""
    ProdFinal{T}

The `ProdFinal` is a wrapper around a distribution. By passing it as a message along an edge of the graph the corresponding marginal is calculated as the distribution of the `ProdFinal`. 
In a sense, the `ProdFinal` ignores any further prod with any other distribution for calculating the marginal and only check for variate types of two distributions. Trying to prod two instances of `ProdFinal` will result in an error.
Note: `ProdFinal` is not a prod strategy, as opposed to `ProdAnalytical` and `ProdGeneric`.

See also: [`BIFM`]
"""
struct ProdFinal{T} <: AbstractProdConstraint
    dist::T
end

ProdFinal(prod::ProdFinal) = prod

Base.show(io::IO, prod::ProdFinal) = print(io, "ProdFinal(", getdist(prod), ")")

getdist(dist::ProdFinal) = dist.dist

to_marginal(dist::ProdFinal) = getdist(dist)

MacroHelpers.@proxy_methods ProdFinal getdist [
    Distributions.mean,
    Distributions.median,
    Distributions.mode,
    Distributions.shape,
    Distributions.scale,
    Distributions.rate,
    Distributions.var,
    Distributions.std,
    Distributions.cov,
    Distributions.invcov,
    Distributions.logdetcov,
    Distributions.entropy,
    Distributions.params,
    Base.precision,
    Base.length,
    Base.ndims,
    Base.size,
    Base.eltype,
    mean_cov,
    mean_var,
    mean_invcov,
    mean_precision,
    weightedmean_cov,
    weightedmean_var,
    weightedmean_invcov,
    weightedmean_precision,
    probvec,
    weightedmean
]

Distributions.mean(fn::Function, prod::ProdFinal) = mean(fn, getdist(prod))

Distributions.pdf(prod::ProdFinal, x)    = Distributions.pdf(getdist(prod), x)
Distributions.logpdf(prod::ProdFinal, x) = Distributions.logpdf(getdist(prod), x)

custom_isapprox(dist1::ProdFinal, dist2::ProdFinal; kwargs...) = custom_isapprox(getdist(dist1), getdist(dist2); kwargs...)
custom_isapprox(dist1::Any, dist2::ProdFinal; kwargs...)       = custom_isapprox(dist1, getdist(dist2); kwargs...)
custom_isapprox(dist1::ProdFinal, dist2::Any; kwargs...)       = custom_isapprox(getdist(dist1), dist2; kwargs...)

# This function is unsafe and uses internal fields of Julia types, but should be used only in tests
convert_eltype(::Type{<:ProdFinal}, ::Type{T}, prod::ProdFinal{D}) where {T, D} = ProdFinal(convert_eltype(D.name.wrapper, T, getdist(prod)))

prod_analytical_rule(::Type{<:ProdFinal}, ::Type) = ProdAnalyticalRuleAvailable()
prod_analytical_rule(::Type, ::Type{<:ProdFinal}) = ProdAnalyticalRuleAvailable()

# product of distribution message with `ProdFinal` always returns the same `ProdFinal` object directly
prod_final_check_variate_types(::Type{T}, ::Type{T}, result) where {T <: Distributions.VariateForm} = result
prod_final_check_variate_types(::Type{T1}, ::Type{T2}, result) where {T1, T2} = error("Different variate types in a prod with `ProdFinal`: $(T1) × $(T2)")

prod(::ProdAnalytical, left::ProdFinal, right) = prod_final_check_variate_types(variate_form(getdist(left)), variate_form(right), left)
prod(::ProdAnalytical, left, right::ProdFinal) = prod_final_check_variate_types(variate_form(left), variate_form(getdist(right)), right)

prod_analytical_rule(::Type{<:ProdFinal}, ::Type{<:ProdFinal}) = ProdAnalyticalRuleAvailable()

prod(::ProdAnalytical, left::ProdFinal, right::ProdFinal) = error("Invalid product: `ProdFinal` × `ProdFinal`")

export ProdPreserveParametrisation, ProdBestSuitableParametrisation
export vague
export mean, median, mode, shape, scale, rate, var, std, cov, invcov, entropy, pdf, logpdf, logdetcov
export weightedmean, probvec, logmean, meanlogmean, inversemean, mirroredlogmean, loggammamean
export variate_form, value_support, promote_variate_type

import Distributions: mean, median, mode, shape, scale, rate, var, std, cov, invcov, entropy, pdf, logpdf, logdetcov

import Base: prod


"""
    ProdPreserveParametrisation

`ProdPreserveParametrisation` is one of the strategies for `prod` function. This strategy assumes the same output factorisation (if possible).
Should be used mostly to preserve distribution parametrisation across a model.

See also: [`prod`](@ref), [`ProdBestSuitableParametrisation`](@ref), [`ProdExpectationMaximisation`](@ref)
"""
struct ProdPreserveParametrisation end

"""
    ProdBestSuitableParametrisation

`ProdBestSuitableParametrisation` is one of the strategies for `prod` function. This strategy does not make any assumptions about output factorisation.
Can be used to speedup computations in some cases.

See also: [`prod`](@ref), [`ProdPreserveParametrisation`](@ref), [`ProdExpectationMaximisation`](@ref)
"""
struct ProdBestSuitableParametrisation end

"""
    ProdExpectationMaximisation

`ProdExpectationMaximisation` is one of the strategies for `prod` function. This strategy assumes that the output of prod is a point mass distribution.

See also: [`prod`](@ref), [`ProdPreserveParametrisation`](@ref), [`ProdBestSuitableParametrisation`](@ref)
"""
struct ProdExpectationMaximisation end

"""
    prod(strategy, left, right)

`prod` function is used to find a product of two probability distrubution over same variable (e.g. 𝓝(x|μ_1, σ_1) × 𝓝(x|μ_2, σ_2)).
There are two strategies for prod function: `ProdPreserveParametrisation` and `ProdBestSuitableParametrisation`.

# Examples:
```jldoctest
using ReactiveMP

product = prod(ProdPreserveParametrisation(), NormalMeanVariance(-1.0, 1.0), NormalMeanVariance(1.0, 1.0))

mean(product), var(product)

# output
(0.0, 0.5)
```

See also: [`default_prod_strategy`](@ref), [`ProdPreserveParametrisation`](@ref), [`ProdBestSuitableParametrisation`](@ref)
"""
prod(::ProdBestSuitableParametrisation, left, right) = prod(ProdPreserveParametrisation(), left, right)

prod(::ProdPreserveParametrisation, left::Missing, right) = right
prod(::ProdPreserveParametrisation, left, right::Missing) = left
prod(::ProdPreserveParametrisation, ::Missing, ::Missing) = missing


"""
    vague(distribution_type, [ dims... ])

`vague` function returns uninformative probability distribution of a given type and can be used to set an uninformative priors in a model.
"""
function vague end

probvec(something)         = error("Probability vector function probvec() is not defined for $(something)")
weightedmean(something)    = error("Weighted mean is not defined for $(something)")
inversemean(something)     = error("Inverse expectation is not defined for $(something)")
logmean(something)         = error("Logarithmic expectation is not defined for $(something)")
meanlogmean(something)     = error("xlog(x) expectation is not defined for $(something)")
mirroredlogmean(something) = error("Mirrored Logarithmic expectation is not defined for $(something)")
loggammamean(something)    = error("E[logГ(x)] is not defined for $(something)")

"""
    variate_form(distribution_or_type)

Returns the `VariateForm` sub-type (defined in `Distributions.jl`):

- `Univariate`, a scalar number
- `Multivariate`, a numeric vector
- `Matrixvariate`, a numeric matrix
"""
variate_form(::Distribution{F, S})            where { F <: VariateForm, S <: ValueSupport } = F
variate_form(::Type{ <: Distribution{F, S} }) where { F <: VariateForm, S <: ValueSupport } = F

"""
    value_support(distribution_or_type)

Returns the `ValueSupport` sub-type (defined in `Distributions.jl`):

- `Discrete`, samples take discrete values
- `Continuous`, samples take continuous real values
"""
value_support(::Distribution{F, S})            where { F <: VariateForm, S <: ValueSupport } = S
value_support(::Type{ <: Distribution{F, S} }) where { F <: VariateForm, S <: ValueSupport } = S


"""
    promote_variate_type(::Type{ <: VariateForm }, distribution_type)

Promotes a `distribution_type` to be of the specified variate form (if possible)
"""
function promote_variate_type end

promote_variate_type(::D, T)         where { D <: Distribution } = promote_variate_type(variate_form(D), T)
promote_variate_type(::Type{ D }, T) where { D <: Distribution } = promote_variate_type(variate_form(D), T)
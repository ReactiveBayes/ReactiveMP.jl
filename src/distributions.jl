export ProdPreserveParametrisation, ProdBestSuitableParametrisation
export default_prod_strategy, vague
export mean, median, mode, var, std, cov, invcov, entropy, pdf, logpdf
export logmean, inversemean, mirroredlogmean

import Distributions: mean, median, mode, var, std, cov, invcov, entropy, pdf, logpdf

import Base: prod


"""
    ProdPreserveParametrisation

`ProdPreserveParametrisation` is one of the strategies for `prod` function. This strategy assumes the same output factorisation (if possible).
Should be used mostly to preserve distribution parametrisation across a model.

See also: [`prod`](@ref), [`default_prod_strategy`](@ref), [`ProdBestSuitableParametrisation`](@ref)
"""
struct ProdPreserveParametrisation end

"""
    ProdBestSuitableParametrisation

`ProdBestSuitableParametrisation` is one of the strategies for `prod` function. This strategy does not make any assumptions about output factorisation.
Can be used to speedup computations in some cases.

See also: [`prod`](@ref), [`default_prod_strategy`](@ref), [`ProdPreserveParametrisation`](@ref)
"""
struct ProdBestSuitableParametrisation end

"""
    default_prod_strategy()

Returns a default prod strategy.

See also: [`prod`](@ref), [`ProdPreserveParametrisation`](@ref), [`ProdBestSuitableParametrisation`](@ref)
"""
default_prod_strategy() = ProdPreserveParametrisation()

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

function logmean end

logmean(something) = log(mean(something))

function inversemean end

inversemean(something) = inv(mean(something))

function mirroredlogmean end
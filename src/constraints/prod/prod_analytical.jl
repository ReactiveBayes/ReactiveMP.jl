export ProdAnalytical

import Base: prod

"""
    ProdAnalytical

`ProdAnalytical` is one of the strategies for `prod` function. This strategy uses analytical prod methods but does not constraint a prod to be in any specific form.
It fails if no analytical rules is available, use `ProdGeneric` prod strategy to fallback to approximation methods.

See also: [`prod`](@ref), [`ProdPreserveType`](@ref), [`ProdGeneric`](@ref)
"""
struct ProdAnalytical end

"""
    prod(strategy, left, right)

`prod` function is used to find a product of two probability distrubution over same variable (e.g. 𝓝(x|μ_1, σ_1) × 𝓝(x|μ_2, σ_2)).
There are multiple strategies for prod function, e.g. `ProdAnalytical`, `ProdGeneric` or `ProdPreserveType`.

# Examples:
```jldoctest
using ReactiveMP

product = prod(ProdAnalytical(), NormalMeanVariance(-1.0, 1.0), NormalMeanVariance(1.0, 1.0))

mean(product), var(product)

# output
(0.0, 0.5)
```

See also: [`prod_analytical_rule`](@ref), [`ProdAnalytical`](@ref), [`ProdGeneric`](@ref)
"""
prod(::ProdAnalytical, left, right) = error("No analytical rule available to compute a product of distributions $(left) and $(right).")

prod(::ProdAnalytical, ::Missing, right)     = right
prod(::ProdAnalytical, left, ::Missing)      = left
prod(::ProdAnalytical, ::Missing, ::Missing) = missing
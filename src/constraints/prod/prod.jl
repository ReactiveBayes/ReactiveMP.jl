export ProdAnalyticalRuleAvailable, ProdAnalyticalRuleUnknown
export prod_analytical_rule

import Base: prod

struct ProdAnalyticalRuleAvailable end
struct ProdAnalyticalRuleUnknown end

"""
    prod_analytical_rule(::Type, ::Type)

Returns either `ProdAnalyticalRuleAvailable` or `ProdAnalyticalRuleUnknown` for two given distribution types.
Returns `ProdAnalyticalRuleUnknown` by default.

See also: [`prod`](@ref), [`ProdAnalytical`](@ref), [`ProdGeneric`](@ref)
"""
prod_analytical_rule(::Type, ::Type) = ProdAnalyticalRuleUnknown()

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
prod(_, ::Missing, right)     = right
prod(_, left, ::Missing)      = left
prod(_, ::Missing, ::Missing) = missing
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
export ProdAnalyticalRuleAvailable, ProdAnalyticalRuleUnknown
export prod_analytical_rule

import Base: show, prod
import Distributions

struct ProdAnalyticalRuleAvailable end
struct ProdAnalyticalRuleUnknown end

abstract type AbstractProdConstraint end

"""
    prod_analytical_rule(::Type, ::Type)

Returns either `ProdAnalyticalRuleAvailable` or `ProdAnalyticalRuleUnknown` for two given distribution types.
Returns `ProdAnalyticalRuleUnknown` by default.

See also: [`prod`](@ref), [`ProdAnalytical`](@ref), [`ProdGeneric`](@ref)
"""
prod_analytical_rule(::Type, ::Type) = ProdAnalyticalRuleUnknown()


"""
    resolve_prod_constraint(left, right)

Given two product constraints returns a single one that has a higher priority (if possible).

See also: [`prod`](@ref), [`ProdAnalytical`](@ref), [`ProdGeneric`](@ref)
"""
function resolve_prod_constraint end
export ProdAnalyticalRuleAvailable, ProdAnalyticalRuleUnknown
export prod_analytical_rule

struct ProdAnalyticalRuleAvailable end
struct ProdAnalyticalRuleUnknown end

prod_analytical_rule(::Type, ::Type) = ProdAnalyticalRuleUnknown()
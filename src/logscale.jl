# Log scales, tracked on messages and marginals when a graph is activated with `logscales = true`.
# A rule declares the log scale of its message (`MessagePassingRulesBase`'s `logscale` keyword);
# the engine carries it and combines it through products (`compute_product_of_two_messages`).

import MessagePassingRulesBase: getlogscale, UndefinedLogScale

export getlogscale

# The log scale of a message that no rule computed: an initial one.
const INITIAL_LOGSCALE = UndefinedLogScale(:initial)
# The log scale of a message a rule fallback computed.
const FALLBACK_LOGSCALE = UndefinedLogScale(:fallback)
# The log scale of a product a form constraint changed.
const FORM_CONSTRAINT_LOGSCALE = UndefinedLogScale(:form_constraint)

tracked_logscale(logscale) = logscale
tracked_logscale(::Nothing) = throw(
    ArgumentError("log scales are not tracked; activate the graph with `logscales = true` (RxInfer's `infer(...; logscales = true)`)"),
)

# The log scale of the product of two messages, from their data and log scales. A `missing`
# side leaves the other side's; `nothing` on either side means log scales are not tracked, and
# the product has none either; an undefined one propagates; a pair with no `compute_logscale`
# gives an undefined one. The types decide every branch at compile time.
function product_logscale(new, left, right, left_logscale, right_logscale)
    left isa Missing && return right isa Missing ? nothing : right_logscale
    right isa Missing && return left_logscale
    (left_logscale === nothing || right_logscale === nothing) && return nothing
    left_logscale isa UndefinedLogScale && return left_logscale
    right_logscale isa UndefinedLogScale && return right_logscale
    applicable(BayesBase.compute_logscale, new, left, right) ||
        return UndefinedLogScale(:no_compute_logscale, (typeof(left), typeof(right)))
    return left_logscale + right_logscale + BayesBase.compute_logscale(new, left, right)
end

# A form constraint that changed the product, returned something other than what it was given,
# makes the product's log scale no longer its own. One that returned its input unchanged, such as
# a check that the form is supported, keeps it.
constrained_logscale(unconstrained, constrained, logscale::Real) =
    constrained === unconstrained ? logscale : FORM_CONSTRAINT_LOGSCALE
constrained_logscale(unconstrained, constrained, logscale) = logscale

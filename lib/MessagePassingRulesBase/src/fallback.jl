"""
    NodeFunctionRuleFallback(extract = mean)
    (fallback::NodeFunctionRuleFallback)(node, target, args::RuleArgs)

A rule fallback: the message a stochastic node sends when no rule matches, computed from its
[`nodefunction`](@ref), the log-density its declaration defines. The message towards `out` or
another interface is the node's log-density in that interface, every other input, a message or
a marginal, collapsed to a point by `extract`. It is a [`NodeFunctionLogPdf`](@ref), unnormalised,
which a form constraint or a product with a proper distribution turns into one. Its log scale is
undefined.

# Arguments
- `extract`: the function collapsing each input to a point. Default: `mean`.

# Returns
Called as `fallback(node, target, args)`: a [`NodeFunctionLogPdf`](@ref), or `nothing` where the
node function does not apply: a deterministic node, a node with groups, a member of a group, or
an input that is a joint.

An engine consults a fallback only when resolution finds no rule, so an error inside a rule is
never turned into a fallback; ReactiveMP takes it as the activation option `rulefallback`. A
fallback of one's own is any callable of `(node, target, args)` returning a message or `nothing`.

# Examples

```julia
fallback = NodeFunctionRuleFallback()
message = fallback(NormalMeanVariance, MessagePassingRulesBase.Target(:out),
                   MessagePassingRulesBase.RuleArgs(m = (μ = PointMass(0.0), v = PointMass(1.0))))
logpdf(message, 0.5)   # logpdf(NormalMeanVariance(0.0, 1.0), 0.5)
```
"""
struct NodeFunctionRuleFallback{E}
    extract::E
end

NodeFunctionRuleFallback() = NodeFunctionRuleFallback(mean)

"""
    NodeFunctionLogPdf(logpdf)

The message a [`NodeFunctionRuleFallback`](@ref) computes: an unnormalised log-density, supported
everywhere. `BayesBase.logpdf(d, x)` and `d(x)` are both `logpdf(x)`, and `insupport(d, x)` is
always `true`.
"""
struct NodeFunctionLogPdf{F}
    logpdf::F
end

BayesBase.logpdf(d::NodeFunctionLogPdf, x) = d.logpdf(x)
BayesBase.insupport(::NodeFunctionLogPdf, x) = true
(d::NodeFunctionLogPdf)(x) = logpdf(d, x)

(fallback::NodeFunctionRuleFallback)(node, target, args) = nothing

function (fallback::NodeFunctionRuleFallback)(node, ::Target{E}, args::RuleArgs) where {E}
    applicable(nodefunction, node) || return nothing
    inputs = (rule_inputs(node, args.m)..., rule_inputs(node, args.q)...)
    all(((key, _),) -> key isa Symbol, inputs) || return nothing
    points = NamedTuple{map(first, inputs)}(map(((_, value),) -> fallback.extract(value), inputs))
    f = nodefunction(node)
    return NodeFunctionLogPdf(x -> f(; points..., (E => x,)...))
end

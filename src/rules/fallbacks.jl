export NodeFunctionRuleFallback

import BayesBase

_mergevals(::Nothing, ::Val{T}) where {T} = T
_mergevals(::Val{T}, ::Nothing) where {T} = T
_mergevals(::Val{L}, ::Val{R}) where {L, R} = (L..., R...)

_extractvalues(f::F, ::Nothing, vals::Tuple) where {F} = f.(vals)
_extractvalues(f::F, vals::Tuple, ::Nothing) where {F} = f.(vals)
_extractvalues(f::F, left::Tuple, right::Tuple) where {F} = (f.(left)..., f.(right)...)

struct FallbackNodeFunctionUnnormalizedLogPdf{F}
    fn::F
end

BayesBase.insupport(f::FallbackNodeFunctionUnnormalizedLogPdf, x) = true
BayesBase.logpdf(f::FallbackNodeFunctionUnnormalizedLogPdf, x) = f.fn(x)

(f::FallbackNodeFunctionUnnormalizedLogPdf)(x) = logpdf(f, x)

"""
    NodeFunctionRuleFallback(extractfn = mean)

A fallback rule for `Stochastic` nodes that uses a specified function (default: mean) to transform messages and marginals into a value. 
It calls the `nodefunction` method to create the message.

When a node is defined with the `@node` macro:

1.	The `nodefunction` typically calls `logpdf` associated with the node's distribution.
2.	The first edge in the `@node` specification is used to evaluate `logpdf` at.
3.	Other edges are used to instantiate the associated distribution object.

```julia
julia> using ReactiveMP, BayesBase, Distributions

julia> struct MyBeta{A, B} <: ContinuousUnivariateDistribution
             a::A
             b::B
       end

julia> BayesBase.logpdf(d::MyBeta, x) = logpdf(Beta(d.a, d.b), x)

julia> BayesBase.insupport(d::MyBeta, x::Real) = true

julia> @node MyBeta Stochastic [out, a, b]

julia> message = @call_rule [fallback = NodeFunctionRuleFallback()] MyBeta(:out, Marginalisation) (m_a = Beta(2, 3), m_b = Beta(3, 2));

julia> logpdf(message, 0.5)
-0.5017644952110732

julia> message = @call_rule [fallback = NodeFunctionRuleFallback(mode)] MyBeta(:out, Marginalisation) (m_a = Beta(2, 3), m_b = Beta(3, 2)); # evaluate at `mode`

julia> logpdf(message, 0.5)
-0.5954237415153454
```
"""
struct NodeFunctionRuleFallback{E}
    extractfn::E
end

function NodeFunctionRuleFallback()
    return NodeFunctionRuleFallback(mean)
end

function (fallback::NodeFunctionRuleFallback)(args...)
    return rulefallback_nodefunction(fallback, args...)
end

function rulefallback_nodefunction(fallback::NodeFunctionRuleFallback, fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node)
    return rulefallback_nodefunction(fallback, sdtype(fform), fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node)
end

function rulefallback_nodefunction(fallback::NodeFunctionRuleFallback, ::Stochastic, fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node)
    vals   = _mergevals(mnames, qnames)
    means  = _extractvalues(fallback.extractfn, messages, marginals)
    kwargs = NamedTuple{vals}(means)
    fn     = ReactiveMP.nodefunction(fform, on; kwargs...)
    return FallbackNodeFunctionUnnormalizedLogPdf(fn), addons
end

function rulefallback_nodefunction(
    fallback::NodeFunctionRuleFallback, ::Deterministic, fform, on::Val{:out}, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node
)
    error("This fallback rule does not work for deterministic nodes. Use `@meta` to define an approximation method for the deterministic node $(fform).")
end

function rulefallback_nodefunction(fallback::NodeFunctionRuleFallback, ::Deterministic, fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node)
    error("This fallback rule does not work for deterministic nodes. Use `@meta` to define an approximation method for a deterministic node $(fform).")
end

import BayesBase

_mergevals(::Nothing, ::Val{T}) where {T} = T
_mergevals(::Val{T}, ::Nothing) where {T} = T
_mergevals(::Val{L}, ::Val{R}) where {L, R} = (L..., R...)

_extractmeans(::Nothing, vals::Tuple) = mean.(vals)
_extractmeans(vals::Tuple, ::Nothing) = mean.(vals)
_extractmeans(left::Tuple, right::Tuple) = (mean.(left)..., mean.(right)...)

struct FallbackNodeFunctionUnnormalizedLogPdf{F}
    fn::F
end

BayesBase.insupport(f::FallbackNodeFunctionUnnormalizedLogPdf, x) = true
BayesBase.logpdf(f::FallbackNodeFunctionUnnormalizedLogPdf, x) = f.fn(x)

(f::FallbackNodeFunctionUnnormalizedLogPdf)(x) = logpdf(f, x)

function rulefallback_nodefunction(fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node)
    return rulefallback_nodefunction(sdtype(fform), fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node)
end

function rulefallback_nodefunction(::Stochastic, fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node)
    vals   = _mergevals(mnames, qnames)
    means  = _extractmeans(messages, marginals)
    kwargs = NamedTuple{vals}(means)
    fn     = ReactiveMP.nodefunction(fform, on; kwargs...)
    return FallbackNodeFunctionUnnormalizedLogPdf(fn), addons
end

function rulefallback_nodefunction(::Deterministic, fform, on::Val{:out}, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node)
    error("This fallback rule does not work for deterministic nodes. Use `@meta` to define an approximation method for the deterministic node $(fform).")
end

function rulefallback_nodefunction(::Deterministic, fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node)
    error("This fallback rule does not work for deterministic nodes. Use `@meta` to define an approximation method for a deterministic node $(fform).")
end
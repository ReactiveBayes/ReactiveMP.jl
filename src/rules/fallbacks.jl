import BayesBase

_mergevals(::Nothing, ::Val{T}) where {T} = T
_mergevals(::Val{T}, ::Nothing) where {T} = T
_mergevals(::Val{L}, ::Val{R}) where {L, R} = (L..., R...)

_extractmeans(::Nothing, vals::Tuple) = mean.(vals)
_extractmeans(vals::Tuple, ::Nothing) = mean.(vals)
_extractmeans(left::Tuple, right::Tuple) = (mean.(left)..., mean.(right)...)

struct UnnormalizedLogPdf{F}
    fn::F
end

BayesBase.insupport(f::UnnormalizedLogPdf, x) = true
BayesBase.logpdf(f::UnnormalizedLogPdf, x) = f.fn(x)

function rulefallback_nodefunction(fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node)
    return rulefallback_nodefunction(sdtype(fform), fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node)
end

function rulefallback_nodefunction(::Stochastic, fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node)
    vals   = _mergevals(mnames, qnames)
    means  = _extractmeans(messages, marginals)
    kwargs = NamedTuple{vals}(means)
    fn     = ReactiveMP.nodefunction(fform, on; kwargs...)
    return UnnormalizedLogPdf(fn), addons
end

function rulefallback_nodefunction(::Deterministic, fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node)
    error("Rule fallback for deterministic nodes is not implemented yet")
end
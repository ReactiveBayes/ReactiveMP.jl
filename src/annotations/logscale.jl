export LogScaleAnnotations, getlogscale

"""
    LogScaleAnnotations <: AbstractAnnotations

Annotation processor that tracks the log-scale factor of a message.
Writes the `:logscale` annotation during rule execution (via `@logscale`) and
merges it across message products by summing the left and right log-scales and
adding the normalisation correction from `compute_logscale`.
"""
struct LogScaleAnnotations <: AbstractAnnotations end

"""
    getlogscale(ann::AnnotationDict)

Return the log-scale value stored in `ann`. Throws `KeyError` if the logscale
annotation has not been set.
"""
getlogscale(ann::AnnotationDict) = get_annotation(ann, :logscale)

"""
    @logscale value

Set the log-scale annotation on the current rule's annotation dict.
Intended to be called inside a `@rule` body. Expands to
`annotate!(getannotations(), :logscale, value)`.
"""
macro logscale(value)
    return esc(:(ReactiveMP.annotate!(getannotations(), :logscale, $(value))))
end

function post_rule_annotations!(
    ::LogScaleAnnotations,
    ann::AnnotationDict,
    mapping,
    messages,
    marginals,
    result,
)
    has_annotation(ann, :logscale) && return nothing
    if isnothing(marginals) && all(m -> m isa PointMass, messages)
        annotate!(ann, :logscale, 0)
    elseif isnothing(messages) && all(m -> m isa PointMass, marginals)
        annotate!(ann, :logscale, 0)
    else
        error(
            "Log-scale annotation has not been set for the message update rule = $(mapping)",
        )
    end
    return nothing
end

function post_product_annotations!(
    ::LogScaleAnnotations,
    merged::AnnotationDict,
    left_ann::AnnotationDict,
    right_ann::AnnotationDict,
    new_dist,
    left_dist,
    right_dist,
)
    left_logscale  = getlogscale(left_ann)
    right_logscale = getlogscale(right_ann)
    new_logscale   = compute_logscale(new_dist, left_dist, right_dist)
    annotate!(merged, :logscale, left_logscale + right_logscale + new_logscale)
    return nothing
end

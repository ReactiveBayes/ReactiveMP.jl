export InputArgumentsAnnotations,
    RuleInputArgumentsRecord,
    ProductInputArgumentsRecord,
    get_rule_input_arguments,
    AddonMemory

"""
    RuleInputArgumentsRecord

Stores the inputs and result of a single message update rule execution: the
`MessageMapping`, the incoming messages tuple, the incoming marginals tuple, and
the computed result distribution.
"""
struct RuleInputArgumentsRecord
    mapping
    messages
    marginals
    result
end

"""
    ProductInputArgumentsRecord

Stores the collection of [`RuleInputArgumentsRecord`](@ref) objects that were
combined during one or more message products. Each element corresponds to one
rule execution that contributed to the product.
"""
struct ProductInputArgumentsRecord
    mappings::Vector{RuleInputArgumentsRecord}
end

"""
    InputArgumentsAnnotations <: AbstractAnnotations

Annotation processor that records the input arguments and result of each
message update rule execution and propagates them through message products.

After a rule executes, stores a [`RuleInputArgumentsRecord`](@ref) under the
`:rule_input_arguments` key of the annotation dict. During message products,
merges the records from the two sides into a [`ProductInputArgumentsRecord`](@ref).
"""
struct InputArgumentsAnnotations <: AbstractAnnotations end

"""
    get_rule_input_arguments(ann::AnnotationDict)

Return the rule input arguments stored in `ann`. The value is a
[`RuleInputArgumentsRecord`](@ref) when the message came directly from a single
rule execution, or a [`ProductInputArgumentsRecord`](@ref) when it is the result
of one or more message products. Throws `KeyError` if the annotation has not been
set.
"""
get_rule_input_arguments(ann::AnnotationDict) = get_annotation(
    ann, :rule_input_arguments
)

function pre_rule_annotations!(
    ::InputArgumentsAnnotations,
    ann::AnnotationDict,
    mapping,
    messages,
    marginals,
)
    return nothing
end

function post_rule_annotations!(
    ::InputArgumentsAnnotations,
    ann::AnnotationDict,
    mapping,
    messages,
    marginals,
    result,
)
    annotate!(
        ann,
        :rule_input_arguments,
        RuleInputArgumentsRecord(mapping, messages, marginals, result),
    )
    return nothing
end

function _merge_input_arguments(
    left::RuleInputArgumentsRecord, right::RuleInputArgumentsRecord
)
    return ProductInputArgumentsRecord(RuleInputArgumentsRecord[left, right])
end

function _merge_input_arguments(
    left::RuleInputArgumentsRecord, right::ProductInputArgumentsRecord
)
    pushfirst!(right.mappings, left)
    return right
end

function _merge_input_arguments(
    left::ProductInputArgumentsRecord, right::RuleInputArgumentsRecord
)
    push!(left.mappings, right)
    return left
end

function _merge_input_arguments(
    left::ProductInputArgumentsRecord, right::ProductInputArgumentsRecord
)
    append!(left.mappings, right.mappings)
    return left
end

function post_product_annotations!(
    ::InputArgumentsAnnotations,
    merged::AnnotationDict,
    left_ann::AnnotationDict,
    right_ann::AnnotationDict,
    new_dist,
    left_dist,
    right_dist,
)
    left_record  = get_rule_input_arguments(left_ann)
    right_record = get_rule_input_arguments(right_ann)
    annotate!(
        merged,
        :rule_input_arguments,
        _merge_input_arguments(left_record, right_record),
    )
    return nothing
end

function Base.show(io::IO, record::RuleInputArgumentsRecord)
    indent = get(io, :indent, 0)
    pad = ' '^indent
    mapping = record.mapping
    println(io, pad, "Rule input arguments:")
    println(io, pad, "  node:       ", message_mapping_fform(mapping))
    println(io, pad, "  interface:  ", mapping.vtag)
    println(io, pad, "  constraint: ", mapping.vconstraint)
    if !isnothing(mapping.meta)
        println(io, pad, "  meta:       ", mapping.meta)
    end
    if !isnothing(record.messages)
        names = unval(mapping.msgs_names)
        for (name, msg) in zip(names, record.messages)
            println(io, pad, "  msg(", name, ") = ", msg)
        end
    end
    if !isnothing(record.marginals)
        names = unval(mapping.marginals_names)
        for (name, mar) in zip(names, record.marginals)
            println(io, pad, "  q(", name, ") = ", mar)
        end
    end
    print(io, pad, "  result:     ", record.result)
end

function Base.show(io::IO, record::ProductInputArgumentsRecord)
    indent = get(io, :indent, 0)
    pad    = ' '^indent
    println(
        io,
        pad,
        "Product of ",
        length(record.mappings),
        " rule input arguments:",
    )
    inner = IOContext(io, :indent => indent + 4)
    for (i, r) in enumerate(record.mappings)
        println(inner, pad, "  [", i, "]")
        show(inner, r)
        i < length(record.mappings) && println(io)
    end
end

"""
    AddonMemory(args...; kwargs...)

Deprecated: `AddonMemory` has been removed in ReactiveMP v6.
Use [`InputArgumentsAnnotations`](@ref) instead. See the migration guide in the documentation for details.
"""
function AddonMemory(args...; kwargs...)
    error(
        """`AddonMemory` has been removed in ReactiveMP v6 and replaced by `InputArgumentsAnnotations`. """ *
        """See the migration guide in the documentation for details.""",
    )
end

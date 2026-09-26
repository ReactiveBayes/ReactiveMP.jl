export InputArgumentsAnnotations,
    RuleInputArgumentsRecord,
    ProductInputArgumentsRecord,
    get_rule_input_arguments

"""
    RuleInputArgumentsRecord

What one message rule call received and returned, which [`InputArgumentsAnnotations`](@ref)
records on its message.

# Fields

- `mapping`: the [`ReactiveMP.MessageMapping`](@ref), which holds the node, the target and the
  algorithm;
- `messages`: the inbound messages the rule read, a tuple, or `nothing`;
- `marginals`: the marginals the rule read, a tuple, or `nothing`;
- `result`: the message's data.
"""
struct RuleInputArgumentsRecord
    mapping
    messages
    marginals
    result
end

"""
    ProductInputArgumentsRecord

The [`RuleInputArgumentsRecord`](@ref)s of the messages a product was formed from, as a flat
list, however deeply the products nest.

# Fields

- `mappings`: the records, a `Vector{RuleInputArgumentsRecord}`, in the order of the product.
"""
struct ProductInputArgumentsRecord
    mappings::Vector{RuleInputArgumentsRecord}
end

"""
    InputArgumentsAnnotations()

The annotation processor that records what each message rule call received and returned, and
carries the record through the products of messages: a message's provenance, for debugging and
for callbacks. After a rule runs, it stores a [`RuleInputArgumentsRecord`](@ref) under the key
`:rule_input_arguments`; a product merges the two sides' records into a
[`ProductInputArgumentsRecord`](@ref), or keeps the one side's that has one. Read the record with
[`get_rule_input_arguments`](@ref).

It is given to the node's activation, `FactorNodeActivationOptions(; annotations =
(InputArgumentsAnnotations(),))`, and to the variables' `MessageProductContext(; annotations =
(InputArgumentsAnnotations(),))`, so that products carry the records on.
"""
struct InputArgumentsAnnotations <: AbstractAnnotations end

"""
    get_rule_input_arguments(ann::AnnotationDict)

The record [`InputArgumentsAnnotations`](@ref) stored in `ann`: a [`RuleInputArgumentsRecord`](@ref)
for a message a rule computed, or a [`ProductInputArgumentsRecord`](@ref) for a product of such
messages.

# Throws

- `KeyError` when `ann` holds no record.
"""
get_rule_input_arguments(ann::AnnotationDict) =
    get_annotation(ann, :rule_input_arguments)

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
    return ProductInputArgumentsRecord(
        vcat(RuleInputArgumentsRecord[left], right.mappings)
    )
end

function _merge_input_arguments(
        left::ProductInputArgumentsRecord, right::RuleInputArgumentsRecord
    )
    return ProductInputArgumentsRecord(
        vcat(left.mappings, RuleInputArgumentsRecord[right])
    )
end

function _merge_input_arguments(
        left::ProductInputArgumentsRecord, right::ProductInputArgumentsRecord
    )
    return ProductInputArgumentsRecord(vcat(left.mappings, right.mappings))
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
    has_left = has_annotation(left_ann, :rule_input_arguments)
    has_right = has_annotation(right_ann, :rule_input_arguments)
    if has_left && has_right
        left_record = get_rule_input_arguments(left_ann)
        right_record = get_rule_input_arguments(right_ann)
        annotate!(
            merged,
            :rule_input_arguments,
            _merge_input_arguments(left_record, right_record),
        )
    elseif has_left
        annotate!(
            merged, :rule_input_arguments, get_rule_input_arguments(left_ann)
        )
    elseif has_right
        annotate!(
            merged, :rule_input_arguments, get_rule_input_arguments(right_ann)
        )
    end
    return nothing
end

function Base.show(io::IO, record::RuleInputArgumentsRecord)
    indent = get(io, :indent, 0)
    pad = ' '^indent
    mapping = record.mapping
    println(io, pad, "Rule input arguments:")
    println(io, pad, "  node:       ", message_mapping_fform(mapping))
    println(io, pad, "  target:     ", mapping.target)
    if !isnothing(mapping.algorithm)
        println(io, pad, "  algorithm:  ", mapping.algorithm)
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
    return print(io, pad, "  result:     ", record.result)
end

function Base.show(io::IO, record::ProductInputArgumentsRecord)
    indent = get(io, :indent, 0)
    pad = ' '^indent
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
    return
end

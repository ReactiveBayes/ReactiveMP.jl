# The arguments a rule receives, built from the latest messages and marginals a node holds.

import MessagePassingRulesBase: Messages, Marginals, RuleArgs, RuleAnnotations, RuleContext
import Random

function rule_product(left, right)
    result = prod(BayesBase.GenericProd(), left, right)
    return result, BayesBase.compute_logscale(result, left, right)
end

"""
    ReactiveMP.GroupInputs{G, N, K}()

Stands for group `G` among the names of a rule's inputs: its members `K`, in order, of the
`N` the group has, which are consecutive in the inputs. The rule receives the group as one
tuple of length `N` in member order, with `nothing` where a member is not an input.
"""
struct GroupInputs{G, N, K} end

"""
    ReactiveMP.GroupMember(group, index, length)

The label of member `index` of a group of `length` members, before [`ReactiveMP.input_names`](@ref)
folds consecutive members into one [`ReactiveMP.GroupInputs`](@ref).
"""
struct GroupMember
    group::Symbol
    index::Int
    length::Int
end

"""
    ReactiveMP.EmptyGroup(group, length)

The label of a group from which a dependency selects no member, such as `m[:in][!k]` with
one member: it takes no input, and the rule receives a tuple of `length` `nothing`s.
"""
struct EmptyGroup
    group::Symbol
    length::Int
end

"""
    ReactiveMP.input_names(labels)

The names of a rule's inputs, as the `Val` a mapping carries: each label is an interface name,
a cluster's member tuple, or a [`ReactiveMP.GroupMember`](@ref), and the consecutive members of
a group become one [`ReactiveMP.GroupInputs`](@ref). A name may appear only once.
"""
function input_names(labels)
    names = Any[]
    for label in labels
        if label isa GroupMember && !isempty(names) && last(names) isa Vector{GroupMember} && first(last(names)).group === label.group
            push!(last(names), label)
        else
            push!(names, label isa GroupMember ? [label] : label)
        end
    end
    folded = map(names) do name
        name isa EmptyGroup && return GroupInputs{name.group, name.length, ()}()
        name isa Vector{GroupMember} || return name
        indices = Tuple(member.index for member in name)
        issorted(indices) || throw(ArgumentError("the members of `$(first(name).group)` must be in member order, got $(indices)"))
        return GroupInputs{first(name).group, first(name).length, indices}()
    end
    keys = map(input_key, folded)
    allunique(keys) || throw(
        ArgumentError("a rule's inputs name `$(first(k for k in keys if count(==(k), keys) > 1))` more than once; a group's members must be consecutive"),
    )
    return Val{Tuple(folded)}()
end

group_name(::GroupInputs{G}) where {G} = G

# The expression for the value under `name`, reading the inputs from position `i` on, and the
# position after it. A group is a tuple of its length with `nothing` for the members left out.
function input_value(name, i)
    name isa GroupInputs || return :(f(inputs[$i])), i + 1
    G, N, K = typeof(name).parameters
    values = map(1:N) do k
        position = findfirst(==(k), K)
        position === nothing ? :nothing : :(f(inputs[$(i + position - 1)]))
    end
    return Expr(:tuple, values...), i + length(K)
end

input_key(name) = name isa GroupInputs ? group_name(name) : name

"""
    ReactiveMP.rule_messages(f, names, messages)

The `Messages` a rule reads, keyed by interface name, each value `f` of the message. A group
is one tuple under its name (see [`ReactiveMP.GroupInputs`](@ref)).
"""
rule_messages(f::F, ::Nothing, ::Nothing) where {F} = Messages(NamedTuple())

@generated function rule_messages(f::F, ::Val{N}, inputs::Tuple) where {F, N}
    keys, values, i = Symbol[], Any[], 1
    for name in N
        value, i = input_value(name, i)
        push!(keys, input_key(name))
        push!(values, value)
    end
    return :(Messages(NamedTuple{$(Tuple(keys))}($(Expr(:tuple, values...)))))
end

"""
    ReactiveMP.rule_marginals(f, names, marginals)

The `Marginals` a rule reads, each value `f` of the marginal. A marginal keyed by a symbol is
the marginal of one interface; one keyed by a tuple of names is the joint of a cluster; a group
is one tuple under its name (see [`ReactiveMP.GroupInputs`](@ref)).

A joint whose value is a `FactorizedCluster` reaches the rule as its blocks instead: a block
of one member as that member's marginal, `q[:out]`, and a larger one as a joint,
`q[:out, :μ]`. A block of one member of a group stays a joint of that member, `q[((:T, 1),)]`,
since the group's length, which `q[:T]` would need, is not known from the cluster. The labels are in the cluster's type, so this is decided at compile time.
"""
rule_marginals(f::F, ::Nothing, ::Nothing) where {F} = Marginals(NamedTuple())

# Defined before the generated function below, whose generator calls them: a generator sees only
# the methods that existed when it was defined. They are closed; no other method is expected.
# The block labels of a marginal holding a `FactorizedCluster`, or `nothing`.
factorized_blocks(::Type) = nothing
factorized_blocks(::Type{<:Marginal{<:MessagePassingRulesBase.FactorizedCluster{K}}}) where {K} = K

# A block's value is its part of the cluster; its annotations are the joint's.
block_value(::typeof(getdata), marginal, block) = getdata(marginal)[block]
block_value(f::F, marginal, block) where {F} = f(marginal)

# A factorised cluster's blocks must partition the cluster's members, as
# `check_factorized_cluster` requires: every member in exactly one block, each block listing
# its members in the cluster's order. The blocks may come in any order, and a block need not be
# contiguous, since each reaches the rule under its own labels.
function partitions(blocks, members)
    flat = Tuple(Iterators.flatten(blocks))
    length(flat) == length(members) && allunique(flat) && all(in(members), flat) || return false
    return all(block -> issorted(map(member -> findfirst(==(member), members), block)), blocks)
end

@generated function rule_marginals(f::F, ::Val{N}, inputs::Tuple) where {F, N}
    singlekeys, singlevalues, jointkeys, jointvalues, i = Symbol[], Any[], Any[], Any[], 1
    for name in N
        blocks = name isa Tuple ? factorized_blocks(inputs.parameters[i]) : nothing
        if blocks !== nothing
            partitions(blocks, name) || return :(
                throw(
                    ArgumentError(
                        $("the marginal of the cluster $(name) is a FactorizedCluster with the blocks $(blocks), which are not a partition of the cluster's members, each block in the cluster's order"),
                    ),
                )
            )
            for block in blocks
                value = :(block_value(f, inputs[$i], Val($(QuoteNode(block)))))
                isone(length(block)) && only(block) isa Symbol ? (push!(singlekeys, only(block)); push!(singlevalues, value)) : (push!(jointkeys, block); push!(jointvalues, value))
            end
            i += 1
            continue
        end
        value, i = input_value(name, i)
        if name isa Tuple
            push!(jointkeys, name)
            push!(jointvalues, value)
        else
            push!(singlekeys, input_key(name))
            push!(singlevalues, value)
        end
    end
    return :(Marginals(NamedTuple{$(Tuple(singlekeys))}($(Expr(:tuple, singlevalues...))), Val($(Tuple(jointkeys))), $(Expr(:tuple, jointvalues...))))
end


"""
    ReactiveMP.rule_arguments(messages_names, messages, marginals_names, marginals)

The `RuleArgs` of a rule call: the data of the messages and marginals it depends on.
"""
rule_arguments(messages_names, messages, marginals_names, marginals) = RuleArgs(
    rule_messages(getdata, messages_names, messages),
    rule_marginals(getdata, marginals_names, marginals),
)

"""
    ReactiveMP.rule_annotations(messages_names, messages, marginals_names, marginals, out)

The `RuleAnnotations` of a rule call: the annotations the inputs carry, keyed like the
arguments, and `out`, where the rule records its own.
"""
rule_annotations(messages_names, messages, marginals_names, marginals, out) = RuleAnnotations(
    rule_messages(getannotations, messages_names, messages),
    rule_marginals(getannotations, marginals_names, marginals),
    out,
)

has_missing_inputs(::Nothing) = false
has_missing_inputs(inputs::Tuple) = any(ismissing, TupleTools.flatten(getdata.(inputs)))

# Re-exported from `MessagePassingRulesBase`: raised when no rule matches a call. Its message
# lists the near misses.
const RuleNotFoundError = MessagePassingRulesBase.RuleNotFoundError

function resolve_rule(spec)
    spec isa MessagePassingRulesBase.RuleNotFound && throw(RuleNotFoundError(spec))
    return spec
end

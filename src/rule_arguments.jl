# The arguments a rule receives, built from the latest messages and marginals a node holds.

import MessagePassingRulesBase: Messages, Marginals, RuleArgs, RuleAnnotations, RuleContext, RuleLogScales
import Random

"""
    ReactiveMP.GroupInputs{G, N, K}()

Group `G` among the names of a rule's inputs: its members `K`, a tuple of indices in order, out
of the `N` the group has, which are consecutive among the inputs. The rule receives the group as
one tuple of length `N` in member order, with `nothing` where a member is not an input.
"""
struct GroupInputs{G, N, K} end

"""
    ReactiveMP.GroupMember(group::Symbol, index::Int, length::Int)

The label of member `index` of a group of `length` members, before
[`ReactiveMP.input_names`](@ref) folds consecutive members into one
[`ReactiveMP.GroupInputs`](@ref).
"""
struct GroupMember
    group::Symbol
    index::Int
    length::Int
end

"""
    ReactiveMP.EmptyGroup(group::Symbol, length::Int)

The label of a group from which a dependency selects no member, such as `m[:in][!k]` for a group
of one member: it takes no input, and the rule receives a tuple of `length` `nothing`s.
"""
struct EmptyGroup
    group::Symbol
    length::Int
end

"""
    ReactiveMP.input_names(labels) -> Val

The names of a rule's inputs, as the `Val` a mapping carries. Each label is an interface name, a
cluster's member tuple, a [`ReactiveMP.GroupMember`](@ref) or a [`ReactiveMP.EmptyGroup`](@ref);
the consecutive members of a group become one [`ReactiveMP.GroupInputs`](@ref).

# Throws

- `ArgumentError` when a group's members are not in member order, or when a name appears more
  than once, such as a group whose members are not consecutive.
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
    ReactiveMP.rule_messages(f, names, messages) -> Messages

The [`Messages`](@extref MessagePassingRulesBase.Messages) a rule reads, `f` of each message keyed
by its interface name: `getdata` for the rule's arguments, `getannotations` for its annotations.
A group is one tuple under its name (see [`ReactiveMP.GroupInputs`](@ref)). With `names` and
`messages` both `nothing`, it is empty.
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
    ReactiveMP.rule_marginals(f, names, marginals) -> Marginals

The [`Marginals`](@extref MessagePassingRulesBase.Marginals) a rule reads, `f` of each marginal. A
marginal keyed by a symbol is the marginal of one interface, one keyed by a tuple of names the
joint of a cluster, and a group is one tuple under its name (see
[`ReactiveMP.GroupInputs`](@ref)).

A joint whose value is a
[`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster) reaches the rule as its
blocks instead: a block of one interface as that interface's marginal, `q[:out]`, and a larger
one as a joint, `q[:out, :μ]`. A block of one member of a group stays a joint of that member,
`q[((:T, 1),)]`, since the group's length, which `q[:T]` needs, is not known from the cluster.
The blocks are in the cluster's type, so this is decided at compile time.

# Throws

- `ArgumentError` when a `FactorizedCluster`'s blocks do not partition the cluster's members,
  each block in the cluster's order.
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
    ReactiveMP.rule_arguments(messages_names, messages, marginals_names, marginals, logscales::Val)

The [`RuleArgs`](@extref MessagePassingRulesBase.RuleArgs) of a rule call: the data of the messages
and marginals it depends on, keyed as the rule declares them, `args.m[:x]` and `args.q[:x]`
(see [`ReactiveMP.rule_messages`](@ref) and [`ReactiveMP.rule_marginals`](@ref)). With
`logscales = Val(true)` it also holds the log scales the messages arrived with, read as
`args.logscale.m[:x]`; `Val(false)` holds none.
"""
rule_arguments(messages_names, messages, marginals_names, marginals) = RuleArgs(
    rule_messages(getdata, messages_names, messages),
    rule_marginals(getdata, marginals_names, marginals),
)
rule_arguments(messages_names, messages, marginals_names, marginals, ::Val{false}) =
    rule_arguments(messages_names, messages, marginals_names, marginals)
rule_arguments(messages_names, messages, marginals_names, marginals, ::Val{true}) = RuleArgs(
    rule_messages(getdata, messages_names, messages),
    rule_marginals(getdata, marginals_names, marginals),
    RuleLogScales(rule_messages(message_logscale, messages_names, messages)),
)

message_logscale(message::Message) = message.logscale
message_logscale(message) = as_message(message).logscale

"""
    ReactiveMP.rule_annotations(messages_names, messages, marginals_names, marginals, out) -> RuleAnnotations

The [`RuleAnnotations`](@extref MessagePassingRulesBase.RuleAnnotations) of a rule call: the
annotations its inputs carry, keyed like its arguments, and `out`, where the rule records its
own: the message's [`ReactiveMP.AnnotationDict`](@ref), or
[`NoAnnotations`](@extref MessagePassingRulesBase.NoAnnotations) for a marginal rule and an average
energy.
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

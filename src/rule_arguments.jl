# The arguments a rule receives, built from the latest messages and marginals a node holds.

import MessagePassingRulesBase: Messages, Marginals, RuleArgs, RuleAnnotations, RuleContext

"""
    ReactiveMP.rule_messages(f, names, messages)

The `Messages` a rule reads, keyed by interface name, each value `f` of the message.
"""
rule_messages(f::F, ::Nothing, ::Nothing) where {F} = Messages(NamedTuple())
rule_messages(f::F, ::Val{N}, messages::Tuple) where {F, N} = Messages(NamedTuple{N}(map(f, messages)))

"""
    ReactiveMP.rule_marginals(f, names, marginals)

The `Marginals` a rule reads, each value `f` of the marginal. A marginal keyed by a symbol is
the marginal of one interface; one keyed by a tuple of names is the joint of a cluster.
"""
rule_marginals(f::F, ::Nothing, ::Nothing) where {F} = Marginals(NamedTuple())

@generated function rule_marginals(f::F, ::Val{N}, marginals::Tuple) where {F, N}
    singles = [i for i in eachindex(N) if N[i] isa Symbol]
    joints = [i for i in eachindex(N) if N[i] isa Tuple]
    singlekeys = Tuple(N[i] for i in singles)
    jointkeys = Tuple(N[i] for i in joints)
    singlevalues = Expr(:tuple, (:(f(marginals[$i])) for i in singles)...)
    jointvalues = Expr(:tuple, (:(f(marginals[$i])) for i in joints)...)
    return :(Marginals(NamedTuple{$singlekeys}($singlevalues), Val($jointkeys), $jointvalues))
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

"""
    ReactiveMP.RuleNotFoundError

Re-exported from `MessagePassingRulesBase`: raised when no rule matches a call. Its message
lists the near misses.
"""
const RuleNotFoundError = MessagePassingRulesBase.RuleNotFoundError

function resolve_rule(spec)
    spec isa MessagePassingRulesBase.RuleNotFound && throw(RuleNotFoundError(spec))
    return spec
end

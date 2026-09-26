"""
    RuleResult

What a rule called by hand returns: the `call_*` functions ([`call_message_update_rule`](@ref),
[`call_marginal_update_rule`](@ref), [`call_average_energy`](@ref)), their `@call_*` macros, and
the `message_passing_*` functions ([`message_passing_rule`](@ref) and its siblings). It holds the
rule's result together with everything that produced it. Read it with its getters:
- [`getresult`](@ref): the message, the marginal or the average energy;
- [`getlogscale`](@ref): the log scale the rule declares for a message, a number or an
  [`UndefinedLogScale`](@ref); `nothing` for a marginal or an energy;
- [`getrule`](@ref): the [`RuleSpec`](@ref) that ran;
- [`getalgorithm`](@ref), [`getcontext`](@ref), [`getscratch`](@ref): the algorithm value, the
  [`RuleContext`](@ref) and the working memory it ran with;
- [`getarguments`](@ref): its [`RuleArgs`](@ref); [`gettarget`](@ref): its target;
- [`getannotations`](@ref): where it recorded its annotations.

It shows itself as a report in the terminal (`text/plain`), one line per edge of the node, and as
a card with the node drawn in a notebook or in documentation (`text/html`): the inputs, the
result and its log scale, the rule that ran and the other rules for the same target. The
two-argument `show` is compact, `RuleResult(2.0, logscale = 0)`. An engine runs rules through
[`execute_rule`](@ref) without building one.
"""
struct RuleResult{R, L, A, C, S, G, N, T}
    result::R
    logscale::L
    rule::RuleSpec
    algorithm::A
    context::C
    scratch::S
    arguments::G
    annotations::N
    target::T
end

"""
    getresult(r::RuleResult)

The rule's result: the message, the joint marginal or the average energy. For an in-place rule,
the buffer it wrote into.
"""
getresult(r::RuleResult) = r.result

getlogscale(r::RuleResult) = r.logscale

"""
    getrule(r::RuleResult) -> RuleSpec

The [`RuleSpec`](@ref) that ran, which shows its inputs, source, file and line.
"""
getrule(r::RuleResult) = r.rule

"""
    getalgorithm(r::RuleResult)

The algorithm value the rule ran with: the call's, or [`DefaultAlgorithm`](@ref)`()` for a rule a
[`DefaultAlgorithmExtension`](@ref) inherited ([`rule_algorithm`](@ref)).
"""
getalgorithm(r::RuleResult) = r.algorithm

"""
    getcontext(r::RuleResult) -> RuleContext

The [`RuleContext`](@ref) the rule ran with.
"""
getcontext(r::RuleResult) = r.context

"""
    getscratch(r::RuleResult)

The working memory the rule ran with, as it left it; `nothing` for a rule that declares none.
"""
getscratch(r::RuleResult) = r.scratch

"""
    getarguments(r::RuleResult) -> RuleArgs

The [`RuleArgs`](@ref) the rule ran on.
"""
getarguments(r::RuleResult) = r.arguments

"""
    gettarget(r::RuleResult)

The target the rule computed: an interface, a group member, a cluster, or `nothing` for an
average energy.
"""
gettarget(r::RuleResult) = r.target

"""
    getannotations(r::RuleResult)

Where the rule recorded its annotations: the store the call was given as `ann` (the `out` of a
[`RuleAnnotations`](@ref)), or a [`NoAnnotations`](@ref) when it was given none. Read it with
[`getannotation`](@ref). An engine adds methods for its messages and marginals.
"""
getannotations(r::RuleResult) = r.annotations

function Base.show(io::IO, r::RuleResult)
    print(io, "RuleResult(")
    show(IOContext(io, :compact => true), r.result)
    r.logscale === nothing || (print(io, ", logscale = "); show(IOContext(io, :compact => true), r.logscale))
    print(io, ")")
    return nothing
end

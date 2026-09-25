"""
    RuleResult

What a rule called by hand returns ([`call_message_update_rule`](@ref), the `message_passing_*`
functions and their macros): the rule's result together with everything that produced it.
Read it with its getters:
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
result and its log scale, the rule that ran and the other rules for the same target. An engine
runs rules without building one.
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

The rule's result: the message, the marginal or the average energy.
"""
getresult(r::RuleResult) = r.result

getlogscale(r::RuleResult) = r.logscale

"""
    getrule(r::RuleResult)

The [`RuleSpec`](@ref) that ran.
"""
getrule(r::RuleResult) = r.rule

"""
    getalgorithm(r::RuleResult)

The algorithm value the rule ran with.
"""
getalgorithm(r::RuleResult) = r.algorithm

"""
    getcontext(r::RuleResult)

The [`RuleContext`](@ref) the rule ran with.
"""
getcontext(r::RuleResult) = r.context

"""
    getscratch(r::RuleResult)

The working memory the rule ran with, as it left it; `nothing` for a rule that declares none.
"""
getscratch(r::RuleResult) = r.scratch

"""
    getarguments(r::RuleResult)

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

Where the rule recorded its annotations: the store passed as `ann`, or `NoAnnotations()`.
"""
getannotations(r::RuleResult) = r.annotations

function Base.show(io::IO, r::RuleResult)
    print(io, "RuleResult(")
    show(IOContext(io, :compact => true), r.result)
    r.logscale === nothing || (print(io, ", logscale = "); show(IOContext(io, :compact => true), r.logscale))
    print(io, ")")
    return nothing
end

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

An engine runs rules without building one.
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

function result_label(r::RuleResult)
    spec = r.rule
    spec.kind === :average_energy && return "average energy of $(spec.node)"
    spec.kind === :marginal && return "marginal of $(spec.node) over $(r.target)"
    return "message of $(spec.node) towards $(r.target)"
end

logscale_source(::Real) = "declared"
logscale_source(::Function) = "computed from the inputs"
logscale_source(::FromBody) = "computed by the body"
logscale_source(::Nothing) = "not declared"

function logscale_label(r::RuleResult)
    logscale = r.logscale
    logscale isa UndefinedLogScale && return sprint(describe_undefined, logscale) * " (undefined)"
    return string(logscale, "  (", logscale_source(r.rule.logscale), ")")
end

input_labels(m::Messages) = Pair{String, Any}["m[$(repr(key))]" => value for (key, value) in pairs(m.values)]
input_labels(q::Marginals{N, T, J}) where {N, T, J} = Pair{String, Any}[
    ["q[$(repr(key))]" => value for (key, value) in pairs(q.singles)];
    ["q[$(repr(key))]" => value for (key, value) in zip(J, q.joints)]
]

function Base.show(io::IO, ::MIME"text/plain", r::RuleResult)
    compact = IOContext(io, :compact => true, :limit => true)
    println(io, "RuleResult: ", result_label(r))
    print(io, "  result:    ")
    show(compact, r.result)
    if r.logscale !== nothing
        print(io, "\n  logscale:  ", logscale_label(r))
    end
    args = r.arguments
    inputs = [input_labels(args.m); input_labels(args.q)]
    if !isempty(inputs)
        print(io, "\n  inputs:")
        width = maximum(length ∘ first, inputs)
        for (label, value) in inputs
            print(io, "\n    ", rpad(label, width), " = ")
            show(compact, value)
        end
    end
    if args.logscale !== nothing
        print(io, "\n  incoming log scales:")
        for (key, value) in pairs(args.logscale.m.values)
            print(io, "\n    m[", repr(key), "] = ")
            show(compact, value)
        end
    end
    print(io, "\n  algorithm: ")
    show(compact, r.algorithm)
    services = r.rule.services
    if !isempty(services)
        print(io, "\n  services:  ", join(services, ", "))
    end
    r.scratch === nothing || (print(io, "\n  scratch:   "); show(compact, r.scratch))
    print(io, "\n  rule:      ", r.rule.file, ":", r.rule.line)
    return nothing
end

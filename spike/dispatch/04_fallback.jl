# Phase 0 -- the rule-fallback contract, and the ruleset-axis decision (open item #4).
#
# PHASES.md asks for the fallback contract *regardless* of how #4 lands, and states the
# requirement: a missing rule and an exception thrown inside a selected rule must be
# distinguishable, and the exception must propagate rather than trigger fallback.
#
# v6 is asymmetric here and it is not a considered asymmetry:
#   - `rule(...)` RETURNS a `RuleMethodError` sentinel (src/rule.jl:1585), consumed at
#     src/message.jl:704-715, which is what lets `rulefallback` exist;
#   - `marginalrule(...)` THROWS (src/rule.jl:1750-1755), unguarded at src/marginal.jl:306.
# So marginal rules cannot have a fallback at all, for no stated reason.

include("02_rules.jl")

using .Rules
using .Rules.Machinery
using BayesBase, ExponentialFamily, Test, Printf

const RESULTS = String[]
record(args...) = (s = string(args...); push!(RESULTS, s); println(s))

struct FallbackNode end
struct ExplodingNode end

const SPEC_EXPLODES = RuleSpec(
    (output, algo, ctx, args, ann, node, target) -> error("this rule is broken, but it WAS found");
    source = "(args) -> error(...)",
)

Machinery.find_rule(::Type{ExplodingNode}, ::Target{:out}, ::BP, ::RuleArgs) = SPEC_EXPLODES
# FallbackNode deliberately has no rule at all.

# ---------------------------------------------------------------------------
# THE CONTRACT.
#
# Resolution is a separate, total function. `find_rule` returns a `RuleSpec` or a
# `RuleNotFound`; it never throws and it never runs anything. The fallback is consulted on
# the `RuleNotFound` branch only -- which is decided *before* any body runs.
#
# That makes the guarantee structural rather than a discipline: there is no `try` anywhere
# near the body, so an exception from inside a selected rule cannot reach the fallback even
# if someone wanted it to. A try/catch around execution would get this wrong, and would get
# it wrong silently, by turning a broken rule into a missing one.
#
# It applies uniformly to message rules, marginal rules and average energy. v6's asymmetry
# is removed: every one of them resolves first and executes second.
# ---------------------------------------------------------------------------

function rule_with_fallback(node, target, algorithm, args, fallback = nothing)
    spec = find_rule(node, target, algorithm, args)
    if spec isa RuleNotFound
        fallback === nothing && throw(Machinery.RuleNotFoundError(spec))
        return fallback(node, target, algorithm, args)
    end
    # No try/catch: whatever the body throws is the caller's problem, by construction.
    return spec.body(nothing, algorithm, default_context, args, NoAnnotations(), node, target)
end

# ---------------------------------------------------------------------------
# Open item #4 -- the ruleset axis.
#
# The question is whether rules need a scoping mechanism *beyond* `algorithm`, so that a
# downstream package's rules cannot invalidate or shadow another's. Below: a downstream
# package defining its own algorithm, and its rules coexisting with the standard ones for
# the same node and the same edge, selected purely by the algorithm value.
# ---------------------------------------------------------------------------

struct MyDownstreamAlgorithm <: Algorithm end

const SPEC_DOWNSTREAM = RuleSpec(
    (output, algo, ctx, args, ann, node, target) -> PointMass(-42.0);
    source = "a downstream package's own rule for a standard node and edge",
)

Machinery.find_rule(::Type{NormalMeanVariance}, ::Target{:out}, ::MyDownstreamAlgorithm, ::RuleArgs) = SPEC_DOWNSTREAM

function main()
    record("# Phase 0 -- fallback contract and the ruleset axis")
    record("julia = ", VERSION)
    record("")

    args = RuleArgs(m = (μ = PointMass(0.0), v = PointMass(1.0)))

    record("## 1. Missing rule vs broken rule are distinguishable")

    missing_threw = try
        rule_with_fallback(FallbackNode, Target(:out), BP(), args)
        false
    catch err
        record("  no rule       -> ", typeof(err))
        err isa Machinery.RuleNotFoundError
    end

    broken_threw = try
        rule_with_fallback(ExplodingNode, Target(:out), BP(), args)
        false
    catch err
        record("  broken rule   -> ", typeof(err), ": ", sprint(showerror, err))
        err isa ErrorException
    end
    record("")

    record("## 2. A fallback catches the missing rule and NOT the broken one")
    fallback_calls = Ref(0)
    fb = (node, target, algorithm, a) -> (fallback_calls[] += 1; PointMass(0.0))

    rule_with_fallback(FallbackNode, Target(:out), BP(), args, fb)
    after_missing = fallback_calls[]
    record("  fallback invocations after a MISSING rule : ", after_missing)

    escaped = try
        rule_with_fallback(ExplodingNode, Target(:out), BP(), args, fb)
        false
    catch
        true
    end
    record("  fallback invocations after a BROKEN rule  : ", fallback_calls[] - after_missing)
    record("  the broken rule's exception propagated    : ", escaped)
    record("")

    record("## 3. Open item #4 -- the ruleset axis")
    standard   = rule_with_fallback(NormalMeanVariance, Target(:out), BP(), args)
    downstream = rule_with_fallback(NormalMeanVariance, Target(:out), MyDownstreamAlgorithm(), args)
    record("  same node, same edge, under BP                   : ", standard)
    record("  same node, same edge, under a downstream algorithm: ", downstream)
    record("")
    record("  DECISION: defer. A downstream package that wants its own rules for a standard")
    record("  node and edge declares its own algorithm and gets them, with no shadowing and")
    record("  no ambiguity, because the algorithm is part of the signature. The piracy")
    record("  argument for the axis is already dead (DISCUSSION.md §5: the check is")
    record("  permanently vacuous for rules). Nothing in tree needs scoped rule tables, and")
    record("  the fallback contract above is specified independently of the axis, as")
    record("  PHASES.md requires. Revisit only on a concrete use case -- the cost of adding")
    record("  an axis later is a new keyword, not a resurfacing.")
    record("")

    @testset "fallback contract" begin
        @test missing_threw                                  # missing rule -> RuleNotFoundError
        @test broken_threw                                   # broken rule  -> its own exception
        @test after_missing == 1                             # fallback consulted for missing
        @test fallback_calls[] - after_missing == 0          # never consulted for broken
        @test escaped                                        # broken rule's exception propagates
        @test mean(downstream) == -42.0                      # algorithm scopes the rule table
    end

    open(joinpath(@__DIR__, "..", "results", "fallback-julia-$(VERSION).txt"), "w") do io
        println(io, join(RESULTS, "\n"))
    end
    return nothing
end

main()

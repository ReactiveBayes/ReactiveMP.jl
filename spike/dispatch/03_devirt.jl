# Phase 0 -- THE GATE.
#
# PHASES.md asks for numbers, not a pass/fail: `RuleSpec` carries no type parameters by
# decision, so the body is reached through a `::Function` field and an indirect call is
# expected wherever the compiler cannot see which body sits there. What this file owes is
# where that happens and what it costs.
#
# Two ways to measure this wrongly, both of which flatter whatever was built:
#   - a spec constructed inline inside an inlinable `find_rule` is constant-folded away and
#     reports zero for *every* representation;
#   - a call site that can only ever reach one rule measures the best case and hides the
#     indirect call entirely.
# So every figure below is reported for both call-site shapes, and the parameterised
# alternative is measured alongside so the trade has a number if it is ever reopened.

include("02_rules.jl")

using .Rules
using .Rules.Machinery
using BayesBase, ExponentialFamily, LinearAlgebra, Test, Printf

# JET is optional. The committed Manifest is resolved for the 1.10 floor, and JET does not
# precompile from it on 1.13 (MbedTLS_jll source missing). The routing assertions matter more
# than the JET pass, so a missing JET downgrades that one check rather than failing the gate.
const HAS_JET = try
    @eval using JET
    true
catch
    false
end

const RESULTS = String[]
record(args...) = (s = string(args...); push!(RESULTS, s); println(s))

# ---------------------------------------------------------------------------
# A gate node whose body allocates nothing, so that ROUTING overhead is measured on its own
# rather than mixed with whatever the rule body constructs. PLAN.md § Verification requires
# rule bodies to be assessed separately from the routing.
# ---------------------------------------------------------------------------

struct GateNode end
struct GateNodeB end

const GATE_A = RuleSpec((output, algo, ctx, args, ann, node, target) -> mean(args.m[:a]) + mean(args.m[:b]); source = "a+b")
const GATE_B = RuleSpec((output, algo, ctx, args, ann, node, target) -> mean(args.m[:a]) * mean(args.m[:b]); source = "a*b")

const GateArgs = RuleArgs{<:NamedTuple{(:a, :b), <:Tuple{PointMass, PointMass}}, NamedTuple{(), Tuple{}}}

Machinery.find_rule(::Type{GateNode},  ::Target{:out}, ::BP, ::GateArgs) = GATE_A
Machinery.find_rule(::Type{GateNodeB}, ::Target{:out}, ::BP, ::GateArgs) = GATE_B

mkargs(x, y) = RuleArgs(m = (a = PointMass(x), b = PointMass(y)))

# --- monomorphic: the call site can reach exactly one rule -------------------
function mono(args)
    return message_passing_rule(GateNode, Target(:out), BP(), args)
end

# --- polymorphic: the call site can reach two, chosen at run time ------------
function poly(node, args)
    return message_passing_rule(node, Target(:out), BP(), args)
end

# --- the parameterised alternative, for comparison only ---------------------
struct ParamSpec{B, P}
    body::B
    prealloc::P
    inplace::Bool
    pure::Bool
    source::String
end
@noinline pfind(::Type{GateNode},  ::GateArgs) = ParamSpec((args) -> mean(args.m[:a]) + mean(args.m[:b]), nothing, false, true, "a+b")
@noinline pfind(::Type{GateNodeB}, ::GateArgs) = ParamSpec((args) -> mean(args.m[:a]) * mean(args.m[:b]), nothing, false, true, "a*b")
function pmono(args)
    s = pfind(GateNode, args)
    return s.inplace ? s.body(s.prealloc(args), args) : s.body(args)
end
function ppoly(node, args)
    s = pfind(node, args)
    return s.inplace ? s.body(s.prealloc(args), args) : s.body(args)
end

# ---------------------------------------------------------------------------

# Fixed arity, deliberately. A varargs helper that splats (`f(xs...)`) allocates 48 bytes of
# its own on 1.10 and silently lands in every figure -- this harness had that bug, and the
# gate caught it by reporting 48 bytes for a plain field access. Together with the non-const
# global trap (+16) and constant folding (-everything), that is three separate ways to
# mismeasure the same quantity, all of them found here rather than reasoned about.
allocs1(f, x)       = (f(x);       @allocated f(x))
allocs2(f, x, y)    = (f(x, y);    @allocated f(x, y))

rt(f, T) = (r = Base.return_types(f, T); isempty(r) ? "none" : string(r[1]))

# Dedicated entry points: the node form is a compile-time constant, as it is for a real
# factor node whose functional form is fixed at construction.
call_nmv_out(a)  = message_passing_rule(NormalMeanVariance, Target(:out), BP(), a)
call_nmv_mean(a) = message_passing_rule(NormalMeanVariance, Target(:μ), BP(), a, default_context, Annotations())
call_plus_in2(a) = message_passing_rule(+, Target(:in2), BP(), a)

function main()
    record("# Phase 0 devirtualization gate")
    record("julia = ", VERSION, "  (", Sys.MACHINE, ")")
    record("")

    args = mkargs(1.0, 2.0)
    nodes = (GateNode, GateNodeB)
    pick() = nodes[(time_ns() % 2) + 1]   # runtime-unknown, defeats constant folding

    record("## 1. Keyed argument access")
    faccess(a) = mean(a.m[:a]) + mean(a.m[:b])
    record(@sprintf("  args.m[:sym]                       alloc=%-4d inferred=%s",
                    allocs1(faccess, args), rt(faccess, (typeof(args),))))
    fgroup(t) = t[2]
    record(@sprintf("  group member args.q[:p][k]         alloc=%-4d inferred=%s",
                    allocs1(fgroup, (1.0, 2.0, 3.0)), rt(fgroup, (Tuple{Float64, Float64, Float64},))))
    record("")

    record("## 2. Routing through RuleSpec -- adopted representation (no type parameters)")
    record(@sprintf("  monomorphic call site              alloc=%-4d inferred=%s",
                    allocs1(mono, args), rt(mono, (typeof(args),))))
    n = pick()
    record(@sprintf("  polymorphic call site (2 rules)    alloc=%-4d inferred=%s",
                    allocs2(poly, n, args), rt(poly, (Type, typeof(args)))))
    record("")

    record("## 3. The parameterised alternative, for comparison")
    record(@sprintf("  monomorphic call site              alloc=%-4d inferred=%s",
                    allocs1(pmono, args), rt(pmono, (typeof(args),))))
    record(@sprintf("  polymorphic call site (2 rules)    alloc=%-4d inferred=%s",
                    allocs2(ppoly, n, args), rt(ppoly, (Type, typeof(args)))))
    record(@sprintf("  find_rule return type, param-free  %s", rt(Machinery.find_rule, (Type{GateNode}, Target{:out}, BP, typeof(args)))))
    record(@sprintf("  find_rule return type, param'ised  %s", rt(pfind, (Type{GateNode}, typeof(args)))))
    record("")

    record("## 4. JET on the routing machinery")
    nreports = if HAS_JET
        n = length(JET.get_reports(JET.report_call(mono, (typeof(args),))))
        record("  report_call(mono)                  ", n, " report(s)")
        n
    else
        record("  SKIPPED -- JET unavailable in this environment on julia ", VERSION)
        0
    end
    record("")

    record("## 5. Real rules, end to end (totals -- body allocation included)")
    record("   Each gets its own function on purpose. Looping and closing over the node would")
    record("   capture it as a `DataType` rather than `Type{NormalMeanVariance}`, which alone")
    record("   makes the call dynamic and reports `Any` -- a fourth way to mismeasure this.")
    record("   A real factor node has its form fixed, so the dedicated function is the honest")
    record("   model. (`typeof(+)` is immune: a function is already its own singleton type.)")
    for (name, f, a) in (
            ("NormalMeanVariance(:out)", call_nmv_out, RuleArgs(m = (μ = PointMass(0.0), v = PointMass(1.0)))),
            ("NormalMeanVariance(:μ)  ", call_nmv_mean, RuleArgs(m = (out = PointMass(3.0), v = PointMass(1.0)))),
            ("typeof(+)(:in2)         ", call_plus_in2, RuleArgs(m = (out = PointMass(3.0), in1 = PointMass(1.0)))),
        )
        record(@sprintf("  %-24s           alloc=%-4d inferred=%s", name, allocs1(f, a), rt(f, (typeof(a),))))
    end
    record("")

    @testset "devirtualization gate" begin
        @test allocs1(faccess, args) == 0
        @test allocs1(mono, args) == 0
        @test rt(mono, (typeof(args),)) == "Float64"
        @test rt(Machinery.find_rule, (Type{GateNode}, Target{:out}, BP, typeof(args))) == "RuleSpec"
        HAS_JET && @test nreports == 0
    end

    open(joinpath(@__DIR__, "..", "results", "devirt-julia-$(VERSION).txt"), "w") do io
        println(io, join(RESULTS, "\n"))
    end
    return nothing
end

main()

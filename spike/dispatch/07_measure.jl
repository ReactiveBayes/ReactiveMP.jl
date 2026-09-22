# Phase 0 -- "measure, do not just assert": cold first invocation, warm execution,
# allocations, and specialization growth across variadic group sizes and heterogeneous
# input types.
#
# The specialization question is the one with teeth. The design keys rules on the *set* of
# argument names as well as their types, and a variadic group is an N-tuple, so the worry is
# that many key sets and many input types multiply into a large number of compiled
# specializations. This measures that rather than speculating about it.

include("02_rules.jl")

using .Rules
using .Rules.Machinery
using BayesBase, ExponentialFamily, LinearAlgebra, BenchmarkTools, Printf

const RESULTS = String[]
record(args...) = (s = string(args...); push!(RESULTS, s); println(s))

nmv_args(T) = RuleArgs(m = (μ = PointMass(T(0)), v = PointMass(T(1))))
call_nmv(a) = message_passing_rule(NormalMeanVariance, Target(:out), BP(), a)

# variadic group of size N
mix_args(N) = RuleArgs(m = (
    switch = Categorical(fill(1 / N, N)),
    inputs = ntuple(i -> NormalMeanVariance(Float64(i), 1.0), N),
))
call_mix(a) = message_passing_rule(Rules.Mixture{2}, Target(:out), BP(), a)

nspecializations(f, T) = begin
    total = 0
    for m in methods(f)
        for _ in Base.specializations(m)
            total += 1
        end
    end
    total
end

function measure_type(::Type{T}) where {T}
    a = nmv_args(T)
    r = call_nmv(a)
    call_nmv(a)
    return (typeof(r), @allocated call_nmv(a))
end

function main()
    record("# Phase 0 -- cold, warm, allocations, specialization growth")
    record("julia = ", VERSION, "  (", Sys.MACHINE, ")")
    record("")

    record("## 1. Cold first invocation (compile time for one rule, seconds)")
    a64 = nmv_args(Float64)
    # The node, its rule and the call site are all created AT RUN TIME. Defining them ahead
    # of `main` lets `main`'s own compilation absorb the cost, and the figure reads as zero.
    # (It did, at first -- a fifth way to mismeasure something in this file.)
    @eval struct ColdNodeX end
    @eval Machinery.find_rule(::Type{ColdNodeX}, ::Target{:out}, ::BP, ::RuleArgs) = RuleSpec(
        (output, algo, ctx, args, ann, node, target) -> NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v]))
    )
    coldcall = @eval (a) -> message_passing_rule(ColdNodeX, Target(:out), BP(), a)
    cold      = @elapsed Base.invokelatest(coldcall, a64)
    warm_once = @elapsed Base.invokelatest(coldcall, a64)
    record(@sprintf("   first call, including compilation : %.4f s", cold))
    record(@sprintf("   second call                       : %.8f s", warm_once))
    record("   Node, rule and call site are all created at run time, so the first call pays")
    record("   for compilation rather than having it absorbed by `main`.")
    record("")

    record("## 2. Warm execution")
    b = @benchmark call_nmv($a64)
    record(@sprintf("   minimum %8.2f ns   median %8.2f ns   allocations %d",
                    minimum(b).time, median(b).time, minimum(b).memory))
    record("")

    record("## 3. Heterogeneous input types -- does the rule still specialise cleanly?")
    # Behind a function barrier on purpose: measuring inside the loop makes `a` a variable of
    # changing type, the call goes dynamic, and every row reports a spurious allocation.
    for T in (Float64, Float32, BigFloat)
        r, al = measure_type(T)
        record(@sprintf("   %-8s -> %-34s alloc=%d", T, string(r), al))
    end
    record("   Number types propagate through without a widening. That is the property that")
    record("   makes ForwardDiff.Dual work through a rule, which PLAN.md § Testing makes the")
    record("   default-on `check_type_promotion`.")
    record("")

    record("## 4. Specialization growth across variadic group sizes")
    before = nspecializations(Machinery.message_passing_rule, nothing)
    sizes = (2, 3, 4, 5, 8, 16)
    for N in sizes
        call_mix(mix_args(N))
    end
    after = nspecializations(Machinery.message_passing_rule, nothing)
    record("   group sizes exercised            : ", collect(sizes))
    record("   message_passing_rule specializations before : ", before)
    record("   message_passing_rule specializations after  : ", after)
    record("   added                                       : ", after - before)
    record("")
    record("   Each distinct group size is a distinct tuple type, so it is a distinct")
    record("   specialization -- exactly as `ManyOf{N,T}` is today. This is a property of")
    record("   the tuple, not of the new design, and it is the price of a statically known")
    record("   group arity. PLAN.md § Dependencies already requires that arity be static for")
    record("   the same reason.")
    record("")

    record("## 5. Combined: group size x element type")
    combos = 0
    for N in (2, 3), T in (Float64, Float32)
        a = RuleArgs(m = (
            switch = Categorical(fill(1 / N, N)),
            inputs = ntuple(i -> NormalMeanVariance(T(i), T(1)), N),
        ))
        call_mix(a)
        combos += 1
    end
    record("   combinations exercised : ", combos)
    record("   specializations now    : ", nspecializations(Machinery.message_passing_rule, nothing))
    record("   The growth is multiplicative in (group size x element type), which is worth")
    record("   knowing before Phase 5 ports 490 rules. It is not new -- v6 has it too -- but")
    record("   it is the number to watch if compile time becomes the complaint.")

    open(joinpath(@__DIR__, "..", "results", "measure-julia-$(VERSION).txt"), "w") do io
        println(io, join(RESULTS, "\n"))
    end
    return nothing
end

main()

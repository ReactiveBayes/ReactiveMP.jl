# Phase 0 -- the two hard context services, as standalone calls (open item #12).
#
# No graph construction, no Rocket, no engine. PHASES.md names both:
#   (a) the mixture switch rule, which needs normalised-product information AND the
#       incoming log scales;
#   (b) a delta rule needing the node function WITH captured parameters and fixed
#       arguments (v6: `FixedArguments.fix`, delta.jl:184) -- not a node type.
#
# PLAN.md open item #12 says these are "currently just names". The point of this file is to
# turn them into signatures.

include("02_rules.jl")

using .Rules
using .Rules.Machinery
using BayesBase, ExponentialFamily, Distributions, LinearAlgebra, Random, Test, Printf

const RESULTS = String[]
record(args...) = (s = string(args...); push!(RESULTS, s); println(s))

# ===========================================================================
# Service (a): product-with-log-scale.
#
# CONTRACT:  (left, right) -> (distribution, logscale::Real)
#
# In v6 `rules/mixture/switch.jl` allocates a throwaway `randomvar` inside the rule to reach
# this machinery -- the single engine leak in the whole rules tree. As a service it is an
# ordinary function with no engine types in its signature at all.
# ===========================================================================

function product_with_logscale(left, right)
    p = prod(GenericProd(), left, right)
    ls = logpdf(left, mean(right)) + logpdf(right, mean(right)) - logpdf(p, mean(right))
    return (p, ls)
end

# ===========================================================================
# FINDING -- incoming annotations are an input, and `args` has nowhere to put them.
#
# The switch rule needs the log scales *carried by the arriving messages*, not only the
# product's. But PLAN.md's container holds the message DATA (`m[:μ]::PointMass` declares the
# distribution type, and the body calls `mean(args.m[:μ])`), while `ann` is an output sink
# the rule writes to. So there is no declared route for an inbound annotation.
#
# Resolution taken here, and recommended: `args` exposes a second, parallel accessor for the
# annotations that arrived with each message, keyed exactly like `m`. Declaration stays
# about data; `args.ann_in[:out]` is how a rule asks for what came with it. It stays out of
# the type-dispatch story, because annotations must not select the mathematical rule.
# ===========================================================================

struct InboundArgs{M <: NamedTuple, Q <: NamedTuple, A <: NamedTuple}
    m::M
    q::Q
    ann_in::A
end

# ===========================================================================
# Service (b): the node function, with parameters captured and arguments fixed.
#
# CONTRACT:  nodefunction(ctx, target) -> a callable of the FREE arguments only.
#
# The delta node's `f` takes all inputs; a backward rule towards `in_k` needs `f` with every
# other input pinned to its current value, leaving one free argument. v6 does this with
# `FixedArguments.fix`. Reimplemented here by hand -- the contract is what matters, not the
# package.
# ===========================================================================

struct FixedAt{P, V}
    position::P
    value::V
end

function fix(f::F, fixed::Tuple) where {F}
    return function (free...)
        args = Vector{Any}(undef, length(fixed) + length(free))
        freei = 1
        for i in eachindex(args)
            j = findfirst(fx -> fx.position == i, fixed)
            if j === nothing
                args[i] = free[freei]; freei += 1
            else
                args[i] = fixed[j].value
            end
        end
        return f(args...)
    end
end

function main()
    record("# Phase 0 -- context services (open item #12)")
    record("julia = ", VERSION)
    record("")

    record("## Service (a): product-with-log-scale")
    record("   contract: (left, right) -> (distribution, logscale::Real)")
    left  = NormalMeanVariance(0.0, 1.0)
    right = NormalMeanVariance(1.0, 2.0)
    p, ls = product_with_logscale(left, right)
    record("   prod(N(0,1), N(1,2)) = ", p)
    record("   logscale             = ", ls)
    record("")
    record("   Called with no graph, no Rocket and no engine types in the signature. That is")
    record("   the whole of what `rules/mixture/switch.jl` reaches an engine for today.")
    record("")

    record("## Service (a'): the switch rule wired through `ctx`")
    ctx = RuleContext(cholesky, Random.default_rng(), (product = product_with_logscale,))
    inputs = (NormalMeanVariance(0.0, 1.0), NormalMeanVariance(3.0, 1.0))
    args = RuleArgs(m = (out = NormalMeanVariance(0.1, 1.0), inputs = inputs))
    switched = message_passing_rule(Rules.Mixture{2}, Target(:switch), BP(), args, ctx)
    record("   Mixture(:switch) -> ", switched)
    record("   the rule asked `service(ctx, :product)`; nothing engine-shaped crossed the")
    record("   boundary, and the service is swappable per call.")
    record("")

    record("## FINDING: incoming annotations have no declared route")
    inbound = InboundArgs(
        (out = NormalMeanVariance(0.1, 1.0), inputs = inputs),
        NamedTuple(),
        (out = (logscale = -1.5,), inputs = ((logscale = -0.2,), (logscale = -0.7,))),
    )
    total = inbound.ann_in.out.logscale + sum(x -> x.logscale, inbound.ann_in.inputs)
    record("   incoming log scales: out=", inbound.ann_in.out.logscale,
           " inputs=", map(x -> x.logscale, inbound.ann_in.inputs), " -> total ", total)
    record("")
    record("   `args` holds message DATA and `ann` is an output sink, so an annotation that")
    record("   ARRIVES with a message has nowhere to go. Recommend a parallel accessor keyed")
    record("   like `m`: `args.ann_in[:out]`. Declaration stays about data; annotations must")
    record("   not participate in dispatch, since they must not select the mathematics.")
    record("")

    record("## Service (b): node function with captured parameters and fixed arguments")
    record("   contract: nodefunction(ctx, target) -> callable of the FREE arguments only")
    g = (x, y, z) -> x + 2y + 3z
    gfixed = fix(g, (FixedAt(1, 10.0), FixedAt(3, 100.0)))     # free argument: y
    record("   f(x,y,z) = x + 2y + 3z, with x=10 and z=100 fixed")
    record("   f_fixed(1.0) = ", gfixed(1.0), "   (expected ", 10.0 + 2 * 1.0 + 300.0, ")")
    record("   f_fixed(2.0) = ", gfixed(2.0))
    record("")
    record("   This is what a delta backward rule towards `in_k` needs: every other input")
    record("   pinned to its current value, one free argument left. The rule receives a")
    record("   CALLABLE, never a node type, so it needs no access to the graph.")
    record("")

    record("## Contracts, as they should be written down in Phase 3")
    record("   product : (left, right)      -> (dist, logscale::Real)")
    record("   nodefn  : (ctx, target)      -> callable of the free arguments")
    record("   linalg  : (matrix)           -> a factorisation object (replaces global cholinv)")
    record("   rng     : ()                 -> an AbstractRNG owned by the caller")
    record("   Rules declare what they need (`ctx = (:product,)`), so a missing service is a")
    record("   named diagnostic rather than a MethodError deep inside a body.")

    @testset "context services" begin
        @test ls isa Real
        @test switched isa Categorical
        @test gfixed(1.0) == 312.0
        @test gfixed(2.0) == 314.0
        @test total ≈ -2.4
    end

    open(joinpath(@__DIR__, "..", "results", "services-julia-$(VERSION).txt"), "w") do io
        println(io, join(RESULTS, "\n"))
    end
    return nothing
end

main()

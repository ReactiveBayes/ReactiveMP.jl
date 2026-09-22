# Phase 0 -- ten representative rules, hand-written in the decided syntax.
#
# Each is shown twice: the surface PLAN.md § Rule surface specifies, in a comment, and the
# hand-lowered form beneath it. The comment is what a user would write; the code is what the
# macro would emit. Reading the pairs side by side is the exit criterion.

module Rules

include("01_machinery.jl")
using .Machinery
using BayesBase, ExponentialFamily, Distributions, LinearAlgebra

export NODES, run_all

# Node markers. For distribution nodes the functional form *is* the distribution type, as in
# v6. `Mixture` and `DeltaFn` are parametric because their interface count is structural.
struct Mixture{N} end
struct DeltaFn{F} end

# ---------------------------------------------------------------------------
# Argument-shape aliases. `args = (m[:μ]::PointMass, m[:v]::PointMass)` constrains the
# RuleArgs type parameters; these aliases just make that readable.
# ---------------------------------------------------------------------------

const NoQ = NamedTuple{(), Tuple{}}
const NoM = NamedTuple{(), Tuple{}}

# messages only
const Msgs{N, T} = RuleArgs{<:NamedTuple{N, <:T}, NoQ}
# marginals only
const Mrgs{N, T} = RuleArgs{NoM, <:NamedTuple{N, <:T}}
# both
const Both{NM, TM, NQ, TQ} = RuleArgs{<:NamedTuple{NM, <:TM}, <:NamedTuple{NQ, <:TQ}}

# ===========================================================================
# 1. The trivial BP case.
#
# @define_message_update_rule(
#     node    = NormalMeanVariance,
#     towards = :out,
#     args    = (m[:μ]::PointMass, m[:v]::PointMass),
#     body    = (args) -> NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v])),
# )
# ===========================================================================

const SPEC_NMV_OUT = RuleSpec(
    (output, algo, ctx, args, ann, node, target) -> NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v]));
    source = "(args) -> NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v]))",
    file = Symbol(@__FILE__), line = @__LINE__,
)

Machinery.find_rule(::Type{NormalMeanVariance}, ::Target{:out}, ::BP, ::Msgs{(:μ, :v), Tuple{PointMass, PointMass}}) = SPEC_NMV_OUT

# ===========================================================================
# 2. The same shape, writing a log scale. `@logscale ls` is gone; the body requests the
#    `ann` slot and calls an ordinary function on it.
#
#     body = (args, ann) -> begin
#         annotate!(ann, :logscale, 0.0)
#         NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v]))
#     end
# ===========================================================================

const SPEC_NMV_MEAN = RuleSpec(
    (output, algo, ctx, args, ann, node, target) -> begin
        annotate!(ann, :logscale, 0.0)
        return NormalMeanVariance(mean(args.m[:out]), mean(args.m[:v]))
    end;
    source = "(args, ann) -> begin annotate!(ann, :logscale, 0.0); ... end",
    file = Symbol(@__FILE__), line = @__LINE__,
)

Machinery.find_rule(::Type{NormalMeanVariance}, ::Target{:μ}, ::BP, ::Msgs{(:out, :v), Tuple{PointMass, PointMass}}) = SPEC_NMV_MEAN

# ===========================================================================
# 3. The arithmetic catch-all -- `meta::Any` in v6, half of the known 27-way ambiguity
#    against the delta catch-all. Here it is an ordinary rule under `BP`, with nothing
#    catch-all about it: the delta rules live under a *different algorithm*, so the two
#    cannot collide. That is the claim this rule exists to test.
# ===========================================================================

const SPEC_PLUS_IN2 = RuleSpec(
    (output, algo, ctx, args, ann, node, target) -> PointMass(mean(args.m[:out]) - mean(args.m[:in1]));
    source = "(args) -> PointMass(mean(args.m[:out]) - mean(args.m[:in1]))",
    file = Symbol(@__FILE__), line = @__LINE__,
)

Machinery.find_rule(::typeof(+), ::Target{:in2}, ::BP, ::Msgs{(:out, :in1), Tuple{PointMass, PointMass}}) = SPEC_PLUS_IN2

# ===========================================================================
# 4. THE CANARY. `NormalMixture((:m, k))`: indexed target + variadic group + aligned
#    dependency, which in v6 needs `where {N}` and `ManyOf` on top of the indexed `on` axis.
#
# @define_message_update_rule(
#     node    = NormalMixture,
#     towards = (:m, k),
#     args    = (q[:out]::Any, q[:switch]::Any, q[:p...]::Any),
#     body    = (args) -> ...,          # uses args.q[:p][k]
# )
#
#    `k` comes off the target, not the group: `index(target)`. The group is reached as
#    `args.q[:p][k]`, which parses as `(q[:p])[k]` and matches runtime access exactly.
# ===========================================================================

struct NormalMixture end

const SPEC_NMIX_M = RuleSpec(
    (output, algo, ctx, args, ann, node, target) -> begin
        k  = index(target)            # macro-injected binding from `towards = (:m, k)`
        pk = args.q[:p][k]
        return NormalMeanPrecision(mean(args.q[:out]), mean(pk))
    end;
    source = "(args) -> NormalMeanPrecision(mean(args.q[:out]), mean(args.q[:p][k]))",
    file = Symbol(@__FILE__), line = @__LINE__,
)

Machinery.find_rule(::Type{NormalMixture}, ::IndexedTarget{:m}, ::VMP, ::Mrgs{(:out, :switch, :p), <:Tuple{Any, Any, Tuple}}) = SPEC_NMIX_M

# ===========================================================================
# 5. Variadic group, `all` selector. In v6 this rule indexes the raw `messages` tuple.
# ===========================================================================

const SPEC_MIX_OUT = RuleSpec(
    (output, algo, ctx, args, ann, node, target) -> begin
        inputs = args.m[:inputs]
        w      = probvec(args.m[:switch])
        return MixtureDistribution(collect(inputs), w)
    end;
    source = "(args) -> MixtureDistribution(collect(args.m[:inputs]), probvec(args.m[:switch]))",
    file = Symbol(@__FILE__), line = @__LINE__,
)

Machinery.find_rule(::Type{<:Mixture}, ::Target{:out}, ::BP, ::Msgs{(:switch, :inputs), <:Tuple{Any, Tuple}}) = SPEC_MIX_OUT

# ===========================================================================
# 6. The single engine leak in the whole rules tree. In v6 this body allocates a throwaway
#    `randomvar` to reach product-with-log-scale machinery. Here it asks the context for the
#    service instead -- open item #12.
# ===========================================================================

const SPEC_MIX_SWITCH = RuleSpec(
    (output, algo, ctx, args, ann, node, target) -> begin
        prod_with_scale = service(ctx, :product)
        inputs = args.m[:inputs]
        out    = args.m[:out]
        scales = map(input -> last(prod_with_scale(out, input)), inputs)
        return Categorical(softmax_(collect(scales)))
    end;
    pure = true,
    source = "(ctx, args) -> ... service(ctx, :product) ...",
    file = Symbol(@__FILE__), line = @__LINE__,
)

softmax_(x) = (e = exp.(x .- maximum(x)); e ./ sum(e))

Machinery.find_rule(::Type{<:Mixture}, ::Target{:switch}, ::BP, ::Msgs{(:out, :inputs), <:Tuple{Any, Tuple}}) = SPEC_MIX_SWITCH

# ===========================================================================
# 7. Delta under Linearization, backward message with no known inverse. Consumes the joint
#    marginal `q[:ins]` and the inbound message on the target edge.
# ===========================================================================

const SPEC_DELTA_IN_LIN = RuleSpec(
    (output, algo, ctx, args, ann, node, target) -> begin
        k = index(target)
        return NormalMeanVariance(mean(args.m[:in]), var(args.m[:in]) + k)
    end;
    source = "(algo, args) -> ... linearize around mean(args.q[:ins]) ...",
    file = Symbol(@__FILE__), line = @__LINE__,
)

Machinery.find_rule(::Type{<:DeltaFn}, ::IndexedTarget{:in}, ::Linearization{Nothing}, ::Both{(:in,), <:Tuple{Any}, (:ins,), <:Tuple{Any}}) = SPEC_DELTA_IN_LIN

# ===========================================================================
# 8. Delta with a KNOWN inverse. Same node, same target, different algorithm value -- and
#    the dependency set differs too: no marginals at all, messages only. In v6 this is a
#    different *layout*; here it is a different `Linearization{I}`.
# ===========================================================================

const SPEC_DELTA_IN_INV = RuleSpec(
    (output, algo, ctx, args, ann, node, target) -> begin
        finv = algo.inverse
        return PointMass(finv(mean(args.m[:out])))
    end;
    source = "(algo, args) -> PointMass(algo.inverse(mean(args.m[:out])))",
    file = Symbol(@__FILE__), line = @__LINE__,
)

Machinery.find_rule(::Type{<:DeltaFn}, ::IndexedTarget{:in}, ::Linearization{<:Function}, ::Msgs{(:out,), <:Tuple{Any}}) = SPEC_DELTA_IN_INV

# ===========================================================================
# 9. A MARGINAL rule over a structural cluster -- `@marginalrule DeltaFn(:ins)` in v6, with
#    `ManyOf{N}`. The cluster is `q[:ins]`; the group arrives as an ordinary Tuple.
# ===========================================================================

const SPEC_DELTA_INS = RuleSpec(
    (output, algo, ctx, args, ann, node, target) -> begin
        ins = args.m[:ins]
        return FactorizedJoint(ins)
    end;
    source = "(args) -> FactorizedJoint(args.m[:ins])",
    file = Symbol(@__FILE__), line = @__LINE__,
)

struct FactorizedJoint{T}
    components::T
end

Machinery.find_rule(::Type{<:DeltaFn}, ::Target{:ins}, ::Unscented, ::Msgs{(:out, :ins), <:Tuple{Any, Tuple}}) = SPEC_DELTA_INS

# ===========================================================================
# 10. IN-PLACE. `preallocate` declares the buffer shape; the body writes into it. The output
#     type is pinned by the ordinary typed lambda parameter, which Julia enforces itself.
#
#     Note the body uses BOTH `output` and `args.m[:out]` -- the earlier design forbade an
#     in-place rule from requesting the inbound message on its own edge, because `m[target]`
#     had to mean one or the other. With `output` as its own slot the ambiguity is gone.
# ===========================================================================

const SPEC_MVN_OUT_INPLACE = RuleSpec(
    (output::MvNormalMeanPrecision, algo, ctx, args, ann, node, target) -> begin
        copyto!(mean(output), mean(args.m[:μ]))
        copyto!(precision(output), precision(args.m[:μ]))
        return output
    end;
    prealloc = (args) -> MvNormalMeanPrecision(
        similar(mean(args.m[:μ])),
        similar(precision(args.m[:μ])),
    ),
    inplace = true,
    source = "(output::MvNormalMeanPrecision, args) -> ...",
    file = Symbol(@__FILE__), line = @__LINE__,
)

Machinery.find_rule(::Type{MvNormalMeanPrecision}, ::Target{:out}, ::BP, ::Msgs{(:μ,), <:Tuple{MvNormalMeanPrecision}}) = SPEC_MVN_OUT_INPLACE

# ===========================================================================
# 11 (bonus). Average energy -- the third generic function, so it is not designed blind.
# ===========================================================================

const SPEC_NMV_ENERGY = RuleSpec(
    (output, algo, ctx, args, ann, node, target) -> begin
        q_out, q_μ, q_v = args.q[:out], args.q[:μ], args.q[:v]
        return (log(2π) + mean(log, q_v) + (var(q_out) + var(q_μ) + abs2(mean(q_out) - mean(q_μ))) / mean(q_v)) / 2
    end;
    source = "(args) -> average energy of NormalMeanVariance",
    file = Symbol(@__FILE__), line = @__LINE__,
)

end # module

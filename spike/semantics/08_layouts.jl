# Phase 0 -- the four delta layouts, expressed as dependency declarations.
#
# THE HYPOTHESIS UNDER TEST (PLAN.md § CVI projection, DISCUSSION.md §3.9b-ii):
# `AbstractDeltaNodeDependenciesLayout` is a bespoke version of the dependency language. If
# it holds, layouts collapse into declarations and `CVIProjection` ships as a weakdep
# extension of the Delta node package. If it does not, the layout half is engine code, node
# packages must not depend on the engine, and CVIProjection needs its own package.
#
# The four layouts, located:
#   DeltaFnDefaultRuleLayout                      src/nodes/predefined/delta/layouts/default.jl:18
#   DeltaFnDefaultKnownInverseRuleLayout          src/nodes/predefined/delta/layouts/default.jl:234
#   CVIApproximationDeltaFnRuleLayout             src/nodes/predefined/delta/layouts/cvi.jl:16
#   CVIProjectionApproximationDeltaFnRuleLayout   ext/ReactiveMPProjectionExt/layout/cvi_projection.jl:35
#
# Each implements or delegates `deltafn_apply_layout` over four slots, applied in a fixed
# order by `activate!` (delta.jl:287-351): q_out -> q_ins -> m_out -> m_in.

using ReactiveMP, Rocket, BayesBase, ExponentialFamily, Test, Printf
import ReactiveMP: DeltaMeta, Linearization, Unscented, deltafn_rule_layout, randomvar,
    factornode, DeltaFnDefaultRuleLayout, DeltaFnDefaultKnownInverseRuleLayout

const RESULTS = String[]
record(args...) = (s = string(args...); push!(RESULTS, s); println(s))

# ---------------------------------------------------------------------------
# The four layouts as declarations, in the vocabulary of PLAN.md § Dependencies as a
# language. `allbutself` is the selector delta already uses (TupleTools.deleteat).
# ---------------------------------------------------------------------------

const DECLARATIONS = (
    default = """
        algorithm = DeltaLinearization / DeltaUnscented
        dependencies = [
            :out      => (m[:ins...],),                    # allbutself is not needed: all ins
            (:in, k)  => (m[:in][k], q[:ins]),             # message on own edge + joint marginal
            :ins      => (m[:out], m[:ins...]),            # the marginal rule over the cluster
        ]""",
    known_inverse = """
        algorithm = DeltaLinearization{HasInverse}
        dependencies = [
            :out      => (m[:ins...],),
            (:in, k)  => (m[:out], m[:ins...] \\ k),        # allbutself; NO marginals at all
            :ins      => (m[:out], m[:ins...]),
        ]""",
    cvi = """
        algorithm = DeltaCVI                               # deleted in Phase 6; reference only
        dependencies = [
            :out      => (q[:ins],),                       # NO inbound message on out
            (:in, k)  => (m[:in][k], q[:ins]),             # delegates to default
        ]""",
    cvi_projection = """
        algorithm = DeltaCVIProjection
        dependencies = [
            :out      => (m[:out], q[:out], q[:ins]),      # the only layout consuming q_out
            (:in, k)  => (m[:in][k], q[:ins]),             # delegates to default
        ]""",
)

function main()
    record("# Phase 0 -- the four delta layouts as dependency declarations")
    record("julia = ", VERSION)
    record("")

    record("## 1. Layout selection is already a function of (method, inverse)")
    f(x) = 2x + 1
    out = randomvar(); x = randomvar()
    node = factornode(f, [(:out, out), (:in, x)], ((1, 2),))
    for (label, meta) in (
            ("Linearization, no inverse ", DeltaMeta(method = Linearization())),
            ("Linearization + inverse   ", DeltaMeta(method = Linearization(), inverse = (y) -> (y - 1) / 2)),
            ("Unscented, no inverse     ", DeltaMeta(method = Unscented())),
            ("Unscented + inverse       ", DeltaMeta(method = Unscented(), inverse = (y) -> (y - 1) / 2)),
        )
        record("   ", label, " -> ", nameof(typeof(deltafn_rule_layout(node, meta))))
    end
    record("")
    record("   Note what this shows: `method` and `layout` are ALREADY two axes, and the")
    record("   layout is a function of (method, inverse) rather than of the method alone.")
    record("   In the new design both collapse into one algorithm VALUE -- the inverse is a")
    record("   field of the algorithm, which is why `Linearization{Nothing}` and")
    record("   `Linearization{<:Function}` select different rules by ordinary dispatch.")
    record("   That is rule 7 vs rule 8 in spike/dispatch/02_rules.jl, and it works.")
    record("")

    record("## 2. The four layouts, written as declarations")
    for k in (:default, :known_inverse, :cvi, :cvi_projection)
        record("   --- ", k, " ---")
        for line in split(getfield(DECLARATIONS, k), "\n")
            record("   ", line)
        end
        record("")
    end

    record("## 3. VERDICT -- the hypothesis HOLDS for input selection, and only for that")
    record("")
    record("   Read across the four, the layouts differ in exactly one respect: which")
    record("   messages and marginals each of the four slots consumes. That is a dependency")
    record("   declaration, and the table above is it. The ~10 engine calls each layout")
    record("   makes are the same calls with different arguments -- repeated wiring, as the")
    record("   hypothesis guessed.")
    record("")
    record("   THREE THINGS DO NOT FIT, and they are not dependency choices:")
    record("")
    record("   (a) STATIC GATING. `with_statics` (default.jl:22-44) wraps every outbound")
    record("       stream in a combineLatest against const/data inputs, so the node WAITS")
    record("       for them, while their values reach the rule out-of-band through the")
    record("       function proxy (FixedArguments.fix, delta.jl:184). Measured in")
    record("       09_execution.jl: 0 emissions before the static arrives, 2 after. A")
    record("       declaration that only names inputs cannot express this. It is execution")
    record("       gating and the new design must carry it as its own concept.")
    record("")
    record("   (b) THE N === 1 COMPILE-TIME BRANCH (default.jl:321-327), substituting")
    record("       `of(Message(nothing, true, true))` for an empty group. A group selector")
    record("       with statically known arity covers this only if the zero-arity case is a")
    record("       declared value rather than an empty tuple that stalls the combineLatest.")
    record("")
    record("   (c) `q_out` ALIASING. `deltafn_apply_layout(::Val{:q_out})` (default.jl:47-64)")
    record("       connects the local marginal straight to the connected variable's marginal")
    record("       stream. That is topology, not a rule input -- nothing computes it.")
    record("")
    record("   CONSEQUENCE FOR THE PLAN: the collapse is real but partial. Dependencies")
    record("   absorb the input selection; statics, empty-group arity and marginal aliasing")
    record("   need explicit support in MessagePassingRulesBase or they land back in the")
    record("   engine. PLAN.md § CVI projection makes CVIProjection-as-an-extension")
    record("   conditional on the collapse, and the condition is met for the rules half --")
    record("   provided those three are lifted out of the layout and into the language.")
    record("")
    record("   Old CVI is a migration reference only: it is deleted in Phase 6, so its")
    record("   declaration above exists to show that the shape fits, not to be built.")

    @testset "layout selection" begin
        @test deltafn_rule_layout(node, DeltaMeta(method = Linearization())) isa DeltaFnDefaultRuleLayout
        @test deltafn_rule_layout(node, DeltaMeta(method = Unscented())) isa DeltaFnDefaultRuleLayout
        @test deltafn_rule_layout(node, DeltaMeta(method = Linearization(), inverse = identity)) isa DeltaFnDefaultKnownInverseRuleLayout
        @test deltafn_rule_layout(node, DeltaMeta(method = Unscented(), inverse = identity)) isa DeltaFnDefaultKnownInverseRuleLayout
    end

    open(joinpath(@__DIR__, "..", "results", "layouts-julia-$(VERSION).txt"), "w") do io
        println(io, join(RESULTS, "\n"))
    end
    return nothing
end

main()

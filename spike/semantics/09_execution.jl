# Phase 0 -- execution semantics, not expressibility.
#
# PHASES.md is explicit about why this file exists: "A declaration can name mathematically
# correct inputs and still produce a graph that stalls or updates in a different order."
# Layouts carry behaviour beyond input choice -- `q_out` aliases the connected variable's
# marginal, static arguments gate execution while their values enter through the function
# proxy, initial values affect stream refresh, self-dependent updates need initialization.
# `dependencies.jl:35-48` records that changing refresh handling changes free-energy
# trajectories and breaks strict FE-monotonicity.
#
# So: run the real v6 engine on minimal cases, and capture EMISSIONS AND NUMBERS. What is
# captured here is the fixture the declarative form must later reproduce -- it is not enough
# for the dependency language to express the inputs.

using ReactiveMP, Rocket, BayesBase, ExponentialFamily, LinearAlgebra, Test, Printf

import ReactiveMP:
    randomvar, datavar, constvar, factornode, activate!, DeltaMeta, Linearization, Unscented,
    RandomVariableActivationOptions, DataVariableActivationOptions, FactorNodeActivationOptions,
    getinterfaces, get_stream_of_outbound_messages, name, getdata, as_message,
    set_initial_message!, deltafn_rule_layout, new_observation!, is_initial, is_clamped

const RESULTS = String[]
record(args...) = (s = string(args...); push!(RESULTS, s); println(s))

"""
Build a delta node over `f`, activate it, and subscribe to every outbound interface stream.
Returns (node, emissions, variables). `emissions` records (edge, value) in the order the
engine actually emits them -- the order is the point, not a detail.
"""
function delta_harness(f, meta, inputs)
    out = randomvar()
    vars = map(inputs) do spec
        spec === :random ? randomvar() : spec
    end
    entries = Tuple{Symbol, Any}[(:out, out)]
    for v in vars
        push!(entries, (:in, v))
    end
    node = factornode(f, entries, (Tuple(1:(length(vars) + 1)),))

    activate!(out, RandomVariableActivationOptions())
    for v in vars
        v isa ReactiveMP.RandomVariable && activate!(v, RandomVariableActivationOptions())
        v isa ReactiveMP.DataVariable   && activate!(v, DataVariableActivationOptions())
    end
    activate!(node, FactorNodeActivationOptions(meta, nothing, nothing, nothing, nothing, nothing))

    emissions = Tuple{Symbol, Any}[]
    for iface in getinterfaces(node)
        nm = name(iface)
        subscribe!(get_stream_of_outbound_messages(iface), (m) -> push!(emissions, (nm, getdata(as_message(m)))))
    end
    return (node, emissions, out, vars)
end

show_emissions(emissions) = for (i, (nm, d)) in enumerate(emissions)
    record(@sprintf("     %d. %-4s -> %s", i, nm, d))
end

function main()
    record("# Phase 0 -- delta execution semantics against the live v6 engine")
    record("julia = ", VERSION)
    record("")

    f(x) = 2x + 1
    finv(y) = (y - 1) / 2

    # -----------------------------------------------------------------------
    record("## 1. DEFAULT layout (Linearization, no inverse)")
    meta1 = DeltaMeta(method = Linearization())
    node1, em1, out1, vars1 = delta_harness(f, meta1, (:random,))
    record("   layout = ", typeof(deltafn_rule_layout(node1, meta1)))
    set_initial_message!(vars1[1], NormalMeanVariance(1.0, 1.0))
    set_initial_message!(out1, NormalMeanVariance(3.0, 2.0))
    record("   emissions in engine order:")
    show_emissions(em1)
    record("   `out` depends on the `ins` messages only; `in` additionally consumes the")
    record("   joint marginal q_ins. Both fire, and the order is out-then-in.")
    record("")

    # -----------------------------------------------------------------------
    record("## 2. KNOWN-INVERSE layout -- same node, same edges, different dependencies")
    meta2 = DeltaMeta(method = Linearization(), inverse = finv)
    node2, em2, out2, vars2 = delta_harness(f, meta2, (:random,))
    record("   layout = ", typeof(deltafn_rule_layout(node2, meta2)))
    set_initial_message!(vars2[1], NormalMeanVariance(1.0, 1.0))
    set_initial_message!(out2, NormalMeanVariance(3.0, 2.0))
    record("   emissions in engine order:")
    show_emissions(em2)
    record("   The backward message is computed from the inbound message on `out` and the")
    record("   other `ins` -- NO marginals at all, where the default layout needed q_ins.")
    record("   Same node, same edges, same four slots: only the dependency sets differ.")
    record("")

    # -----------------------------------------------------------------------
    record("## 3. STATIC INPUT, delivered late -- the part that is NOT a dependency choice")
    g(x, c) = c * x + 1
    meta3 = DeltaMeta(method = Linearization())
    d = datavar()
    node3, em3, out3, vars3 = delta_harness(g, meta3, (:random, d))
    record("   layout = ", typeof(deltafn_rule_layout(node3, meta3)))
    set_initial_message!(vars3[1], NormalMeanVariance(1.0, 1.0))
    set_initial_message!(out3, NormalMeanVariance(3.0, 2.0))
    record("   emissions BEFORE the static input arrives : ", length(em3))
    before = length(em3)
    new_observation!(d, 5.0)
    record("   emissions AFTER  the static input arrives : ", length(em3))
    show_emissions(em3)
    record("")
    record("   This is `with_statics` (delta/layouts/default.jl:22-44) doing its job: the")
    record("   node WAITS, and the static's value reaches the rule out-of-band through the")
    record("   function proxy rather than as a declared input. A dependency declaration that")
    record("   only names inputs cannot express this -- it is execution gating, and the new")
    record("   design has to carry it explicitly.")
    record("")

    # -----------------------------------------------------------------------
    record("## 4. UNSCENTED on the same graph -- the algorithm value alone changes the rule")
    meta4 = DeltaMeta(method = Unscented())
    node4, em4, out4, vars4 = delta_harness(f, meta4, (:random,))
    record("   layout = ", typeof(deltafn_rule_layout(node4, meta4)))
    set_initial_message!(vars4[1], NormalMeanVariance(1.0, 1.0))
    set_initial_message!(out4, NormalMeanVariance(3.0, 2.0))
    show_emissions(em4)
    record("   Same layout as the default case, different rules. Confirms that `method` and")
    record("   `layout` are two axes in v6, and that only one of them is the algorithm.")
    record("")

    # -----------------------------------------------------------------------
    record("## 5. FIXTURES -- what the declarative form must reproduce")
    record("   Not just the values: the emission ORDER and the COUNT. A declaration that")
    record("   names the right inputs and emits twice where v6 emits once is wrong.")
    fixtures = Dict(
        "default"       => (count = length(em1), order = first.(em1)),
        "known-inverse" => (count = length(em2), order = first.(em2)),
        "statics"       => (count = length(em3), before_static = before, order = first.(em3)),
        "unscented"     => (count = length(em4), order = first.(em4)),
    )
    for k in sort(collect(keys(fixtures)))
        record("   ", rpad(k, 15), " => ", fixtures[k])
    end
    record("")

    @testset "delta execution semantics" begin
        @testset "default layout" begin
            @test length(em1) == 2
            @test first.(em1) == [:out, :in]
            @test mean(em1[1][2]) ≈ 3.0        # f(1) = 2*1 + 1
            @test var(em1[1][2])  ≈ 4.0        # 2^2 * 1
        end
        @testset "known inverse" begin
            @test length(em2) == 2
            @test first.(em2) == [:out, :in]
            @test mean(em2[1][2]) ≈ 3.0
        end
        @testset "statics gate execution" begin
            @test before == 0                  # nothing fires until the static arrives
            @test length(em3) > 0              # and everything fires once it does
        end
        @testset "unscented" begin
            @test length(em4) == 2
            @test mean(em4[1][2]) ≈ 3.0 atol = 1e-6
        end
    end

    open(joinpath(@__DIR__, "..", "results", "execution-julia-$(VERSION).txt"), "w") do io
        println(io, join(RESULTS, "\n"))
    end
    return nothing
end

main()

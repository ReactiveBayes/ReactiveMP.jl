# A cluster over a whole group: Delta's joint over its inputs. A cluster is
# written as a tuple of members, `q[:y, :x]` being shorthand for `q[(:y, :x)]`, and inside a
# cluster a group's name means all of its members jointly: `q[(:in,)]`, or `q[:out, :in]`.

@testmodule GroupClusterRules begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: AbstractAlgorithm, annotate!

    struct Joint
        members::Vector{Float64}
    end

    struct DeltaToy end
    @define_factor_node(node = DeltaToy, type = Deterministic, interfaces = [:out, :in...])

    struct ToyDelta <: AbstractAlgorithm end
    @define_dependencies(
        node = DeltaToy,
        algorithm = ToyDelta,
        dependencies = [
            :out => (m[:in...],),
            (:in, k) => (q[(:in,)], m[:in][k]),
        ],
        free_energy_partition = [(:out,), (:in,)],
    )

    # The joint over the group, computed and then consumed.
    @define_marginal_update_rule(
        node = DeltaToy, target = (:in,), algorithm = ToyDelta,
        args = (m[:out]::Float64, m[:in...]::Float64),
        body = (args) -> Joint(collect(args.m[:in]) .+ args.m[:out]),
    )
    @define_message_update_rule(
        node = DeltaToy, target = (:in, k), algorithm = ToyDelta,
        args = (q[(:in,)]::Joint, m[:in][k]::Float64),
        body = (args) -> args.q[(:in,)].members[k] - args.m[:in][k],
    )
    @define_message_update_rule(
        node = DeltaToy, target = :out, algorithm = ToyDelta,
        args = (m[:in...]::Float64,),
        body = (args) -> sum(args.m[:in]),
    )

    # A cluster mixing a single interface and a group.
    struct Mixed <: AbstractAlgorithm end
    @define_dependencies(node = DeltaToy, algorithm = Mixed, dependencies = [:out => (q[:out, :in],)])
    @define_message_update_rule(
        node = DeltaToy, target = :out, algorithm = Mixed,
        args = (q[:out, :in]::Joint,),
        body = (args) -> first(args.q[:out, :in].members),
    )
end

@testitem "group-cluster:rules" tags = [:base] setup = [GroupClusterRules] begin
    using MessagePassingRulesBase: RuleArgs, Marginals, Target, IndexedTarget, ClusterTarget, check_rules
    G = GroupClusterRules

    # The joint's rule, towards the cluster `(:in,)`.
    joint = getresult(message_passing_marginalrule(G.DeltaToy, ClusterTarget((:in,)), G.ToyDelta(), RuleArgs(m = (out = 1.0, in = (2.0, 3.0)))))
    @test joint.members == [3.0, 4.0]

    # A rule consuming it: the engine builds the joint under the key `(:in,)`.
    args = RuleArgs(m = (in = (nothing, 3.0),), q = Marginals(NamedTuple(), Val(((:in,),)), (joint,)))
    @test getresult(message_passing_rule(G.DeltaToy, IndexedTarget(:in, 2), G.ToyDelta(), args)) == 1.0

    mixed = RuleArgs(q = Marginals(NamedTuple(), Val(((:out, :in),)), (G.Joint([7.0, 8.0]),)))
    @test getresult(message_passing_rule(G.DeltaToy, Target(:out), G.Mixed(), mixed)) == 7.0

    # The rules agree with their node and with the dependencies.
    @test isempty(check_rules(G))
end

@testitem "group-cluster:containers" tags = [:base] begin
    using MessagePassingRulesBase: Marginals

    q = Marginals((τ = 1.0,), Val(((:in,), (:y, :x))), ("joint-in", "joint-yx"))
    @test q[(:in,)] === "joint-in"
    # The tuple spelling and the shorthand name the same cluster.
    @test q[(:y, :x)] === q[:y, :x] === "joint-yx"
    @test_throws KeyError q[(:x, :y)]
end

@testitem "group-cluster:dependencies" tags = [:base] setup = [GroupClusterRules] begin
    using MessagePassingRulesBase: dependencies_spec, IndexedTarget, target_dependencies, free_energy_partition
    G = GroupClusterRules

    declaration = dependencies_spec(G.DeltaToy, G.ToyDelta())
    joint, own = target_dependencies(declaration, IndexedTarget(:in, 1))
    @test (joint.container, joint.key) == (:q, (:in,))
    @test free_energy_partition(declaration) == ((:out,), (:in,))

    function failure(ex)
        err = try
            Core.eval(Module(), Expr(:toplevel, :(using MessagePassingRulesBase), :(using MessagePassingRulesBase: AbstractAlgorithm), ex))
            nothing
        catch e
            e isa LoadError ? e.error : e
        end
        return err === nothing ? "" : sprint(showerror, err)
    end
    node(deps) = :(struct N end; @define_factor_node(node = N, type = Deterministic, interfaces = [:out, :μ, :in...], dependencies = $deps))

    # Members stay in interface order, a group included.
    @test failure(node(:([:out => (q[:in, :μ],)]))) |> msg -> contains(msg, "interface order")
    # A one-member cluster of a single interface is just its marginal, with one spelling.
    @test failure(node(:([:out => (q[(:μ,)],)]))) |> msg -> contains(msg, "write `q[:μ]`")
    @test failure(node(:([:out => (q[(:nope,)],)]))) |> msg -> contains(msg, "no interface `nope`")
    @test isempty(failure(node(:([:out => (q[(:in,)],), :μ => (q[(:out, :in)],)]))))
end

@testitem "group-cluster:diagnostics" tags = [:base] begin
    using MessagePassingRulesBase: check_rules
    M = Module()
    definitions = quote
        using MessagePassingRulesBase
        using MessagePassingRulesBase: DefaultAlgorithm
        struct N end
        @define_factor_node(node = N, type = Deterministic, interfaces = [:out, :μ, :in...])
        @define_message_update_rule(node = N, target = :out, algorithm = DefaultAlgorithm, args = (q[(:μ,)]::Any,), body = (args) -> 1)
        @define_message_update_rule(node = N, target = :μ, algorithm = DefaultAlgorithm, args = (q[:in, :out]::Any,), body = (args) -> 1)
        @define_marginal_update_rule(node = N, target = (:μ,), algorithm = DefaultAlgorithm, args = (m[:out]::Any,), body = (args) -> 1)
        @define_message_update_rule(node = N, target = :out, algorithm = DefaultAlgorithm, args = (q[(:in,)]::Any,), body = (args) -> 1)
    end
    # Top level, statement by statement, so `using` takes effect before the macros expand.
    Core.eval(M, Expr(:toplevel, definitions.args...))
    messages = [issue.message for issue in check_rules(M)]
    has(text) = any(m -> contains(m, text), messages)
    @test has("`q[(:μ,)]`: a one-member cluster of a single interface is its marginal; write `q[:μ]`")
    @test has("`q[:in, :out]`: a cluster lists existing interfaces in interface order")
    @test has("`target = (:μ,)`: a one-member cluster of a single interface is its marginal")
    @test length(messages) == 3
end

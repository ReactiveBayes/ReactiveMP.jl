@testmodule PartialGroupNodes begin
    using MessagePassingRulesBase

    # A node with a group, whose rules read joints of some of the group's members with other
    # interfaces, keyed with the members: `(:out, (:T, 1))`.
    struct Tensor end
    @define_factor_node(node = Tensor, type = Stochastic, interfaces = [:out, :in, :T...])

    @define_message_update_rule(
        node = Tensor, target = :in, args = (q[:out, (:T, 1)]::Real, q[:T...]::Any),
        body = (args) -> 10 * args.q[:out, (:T, 1)],
    )
    # A joint of two members, and an energy over a partial joint.
    @define_message_update_rule(node = Tensor, target = :out, args = (q[(:T, 1), (:T, 2)]::Real, q[:in]::Real), body = (args) -> args.q[(:T, 1), (:T, 2)] + args.q[:in])
    @define_average_energy(node = Tensor, args = (q[:out, (:T, 1)]::Real, q[:in]::Real, q[:T...]::Any), body = (args) -> args.q[:out, (:T, 1)] - args.q[:in])
    @define_marginal_update_rule(node = Tensor, target = (:out, (:T, 1)), args = (m[:out]::Real, m[:T...]::Any, q[:in]::Real), body = (args) -> args.m[:out] * args.m[:T][1])
end

@testitem "partial groups:keys" tags = [:base] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: ClusterTarget, cluster_members, Marginals, RuleArgs, canonical_cluster_keys

    # A cluster target and a marginal may name members of a group.
    @test cluster_members(ClusterTarget((:out, (:T, 1)))) == (:out, (:T, 1))
    q = Marginals((in = 2.0,), Val(((:out, (:T, 1)), ((:T, 1), (:T, 2)))), (0.5, 0.25))
    @test q[:out, (:T, 1)] == 0.5 && q[(:out, (:T, 1))] == 0.5
    @test q[(:T, 1), (:T, 2)] == 0.25
    @test q[:in] == 2.0

    # One canonical order for keys, members included, by name and then index; keys of names
    # keep the order they had.
    @test canonical_cluster_keys(((:y, :x), (:a, :b))) == ((:a, :b), (:y, :x))
    @test canonical_cluster_keys(((:out, (:T, 2)), (:out, (:T, 1)), (:in, :out))) == ((:in, :out), (:out, (:T, 1)), (:out, (:T, 2)))
    @test Marginals(NamedTuple(), Val(((:out, (:T, 2)), (:out, (:T, 1)))), (2.0, 1.0))[:out, (:T, 1)] == 1.0
end

@testitem "partial groups:rules" tags = [:base] setup = [PartialGroupNodes] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: check_rules
    N = PartialGroupNodes

    # A rule reads a partial joint by its key, however the call orders the clusters.
    @test getresult(call_message_update_rule(N.Tensor, :in; clusters = ((:out, (:T, 1)) => 0.5,), q = (T = (nothing, 3.0),))) == 5.0
    @test getresult(call_message_update_rule(N.Tensor, :out; clusters = (((:T, 1), (:T, 2)) => 0.25,), q = (in = 1.0,))) == 1.25
    @test getresult(call_average_energy(N.Tensor; clusters = ((:out, (:T, 1)) => 3.0,), q = (in = 1.0, T = (nothing, 2.0)))) == 2.0
    @test getresult(call_marginal_update_rule(N.Tensor, (:out, (:T, 1)); m = (out = 2.0, T = (3.0, nothing)), q = (in = 1.0,))) == 6.0
    # They are well formed.
    @test isempty(check_rules(N))
end

@testitem "partial groups:malformed" tags = [:base] begin
    using MessagePassingRulesBase

    function failure(ex)
        err = try
            Core.eval(Module(), Expr(:toplevel, :(using MessagePassingRulesBase), ex))
            nothing
        catch e
            e isa LoadError ? e.error : e
        end
        return err === nothing ? "" : sprint(showerror, err)
    end
    node = :(struct N end; @define_factor_node(node = N, type = Stochastic, interfaces = [:out, :in, :T...]))
    rule(args) = Expr(:block, node, :(@define_message_update_rule(node = N, target = :in, args = $args, body = (args) -> 1)))
    # A member of what is not a group, and members out of order, are refused by check_rules.
    issues(ex) = (m = Module(); Core.eval(m, Expr(:toplevel, :(using MessagePassingRulesBase), ex)); map(i -> i.message, Core.eval(m, :(MessagePassingRulesBase.check_rules(@__MODULE__)))))
    @test any(contains("is not a group"), issues(rule(:((q[(:out, 1), (:T, 1)]::Any,)))))
    @test any(contains("interface order"), issues(rule(:((q[(:T, 1), :out]::Any,)))))
    @test any(contains("interface order"), issues(rule(:((q[(:T, 2), (:T, 1)]::Any,)))))
    # A member is named by a literal index.
    @test contains(failure(rule(:((q[:out, (:T, j)]::Any,)))), "a group member in a cluster is `(:T, 1)`")
end

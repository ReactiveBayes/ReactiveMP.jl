@testmodule DefaultArgsNodes begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: rule_inputs

    # A node whose rules take whatever inputs the factorisation delivers: each sums the values it
    # is given, weighted by a typed `q(a)`.
    struct Tensor end
    @define_factor_node(node = Tensor, type = Stochastic, interfaces = [:out, :in, :a, :T...])

    total(node, args) = sum(last, rule_inputs(node, args.m); init = 0.0) + sum(last, rule_inputs(node, args.q); init = 0.0)

    @define_message_update_rule(
        node = Tensor, target = :out, args = (default, q[:a]::Float64),
        body = (args) -> args.q[:a] * (total(Tensor, args) - args.q[:a]),
    )
    # A rule with explicit inputs is more specific, and wins where it applies.
    @define_message_update_rule(node = Tensor, target = :out, args = (m[:in]::Float64, q[:a]::Float64), body = (args) -> -1.0)
    @define_message_update_rule(
        node = Tensor, target = (:T, k), args = (default, q[:a]::Float64),
        body = (args) -> k * (total(Tensor, args) - args.q[:a]),
    )
    # A marginal rule over any cluster, which binds the cluster's key.
    @define_marginal_update_rule(
        node = Tensor, target = members, args = (default, q[:a]::Float64),
        body = (args) -> (members, total(Tensor, args) - args.q[:a]),
    )
    @define_average_energy(node = Tensor, args = (default, q[:a]::Float64), body = (args) -> total(Tensor, args))
end

@testitem "default arguments:the inputs" tags = [:base] setup = [DefaultArgsNodes] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: rule_inputs, RuleArgs, Marginals, Messages
    N = DefaultArgsNodes

    # Every input with its key: interfaces by name, a group's members as `(:T, k)`, joints by key.
    q = Marginals((in = 1.0, a = 2.0, T = (nothing, 3.0)), Val(((:out, (:T, 1)),)), (4.0,))
    @test Set(rule_inputs(N.Tensor, q)) == Set([:in => 1.0, :a => 2.0, (:T, 2) => 3.0, (:out, (:T, 1)) => 4.0])
    @test Set(rule_inputs(N.Tensor, Messages((out = 5.0, T = (6.0, nothing, 7.0))))) == Set([:out => 5.0, (:T, 1) => 6.0, (:T, 3) => 7.0])
    # Generated from the container's type, so a rule body walking the pairs is type-stable.
    @test @inferred(rule_inputs(N.Tensor, q)) isa Tuple{Vararg{Pair}}
    @test @inferred(rule_inputs(N.Tensor, Messages((out = 5.0, T = (6.0, nothing, 7.0))))) isa NTuple{3, Pair}
end

@testitem "default arguments:rules" tags = [:base] setup = [DefaultArgsNodes] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: RuleNotFoundError, check_rules, check_rule_ambiguities
    N = DefaultArgsNodes

    # Whatever the factorisation delivers, beside the typed input.
    @test getresult(call_message_update_rule(N.Tensor, :out; m = (T = (1.0, 2.0),), q = (a = 2.0, in = 3.0))) == 2.0 * 6.0
    @test getresult(call_message_update_rule(N.Tensor, :out; clusters = ((:in, (:T, 1)) => 5.0,), q = (a = 1.0, T = (nothing, 1.0)))) == 6.0
    # The typed input is required, and of its type.
    @test_throws RuleNotFoundError getresult(call_message_update_rule(N.Tensor, :out; q = (in = 3.0,)))
    @test_throws RuleNotFoundError getresult(call_message_update_rule(N.Tensor, :out; q = (a = 2, in = 3.0)))
    # A rule with explicit inputs wins.
    @test getresult(call_message_update_rule(N.Tensor, :out; m = (in = 1.0,), q = (a = 2.0,))) == -1.0
    # A group target and its index.
    @test getresult(call_message_update_rule(N.Tensor, (:T, 3); m = (out = 1.0,), q = (a = 2.0, in = 1.0))) == 6.0
    # A marginal rule over any cluster, with its key bound.
    @test getresult(call_marginal_update_rule(N.Tensor, (:out, (:T, 1)); m = (out = 1.0, T = (2.0,)), q = (a = 1.0,))) == ((:out, (:T, 1)), 3.0)
    @test getresult(call_marginal_update_rule(N.Tensor, (:out, :in); m = (out = 1.0, in = 2.0), q = (a = 1.0,))) == ((:out, :in), 3.0)
    @test getresult(call_average_energy(N.Tensor; clusters = ((:out, :in) => 2.0,), q = (a = 1.0,))) == 3.0
    # Checked, and nested rather than ambiguous with the explicit rule.
    @test isempty(check_rules(N))
    @test isempty(check_rule_ambiguities(N))
    @test contains(sprint(show, MIME("text/plain"), which_message_update_rule(N.Tensor, :out; q = (a = 2.0,))), "default")
end

@testitem "default arguments:malformed" tags = [:base] begin
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
    node = :(struct N end; @define_factor_node(node = N, type = Stochastic, interfaces = [:out, :in]))
    rule(args; target = :(:out)) = Expr(:block, node, :(@define_message_update_rule(node = N, target = $target, args = $args, body = (args) -> 1)))
    @test contains(failure(rule(:((default, default)))), "`default` twice")
    # A bare name as a target binds a cluster, so it is a marginal rule's only.
    @test contains(failure(rule(:((default,)); target = :members)), "a bare name is a marginal rule's")
end

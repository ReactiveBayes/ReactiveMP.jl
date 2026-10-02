# `args_check`: a rule's check of its inputs, run before its body, failing with a RuleInputError.

@testmodule CheckedRules begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: AbstractAlgorithm

    struct Scaled <: AbstractAlgorithm
        limit::Float64
    end
    struct Node end
    @define_factor_node(node = Node, type = Stochastic, interfaces = [:out, :y, :p, :in...])

    # A check returning a Bool, reported with its source when false.
    @define_message_update_rule(node = Node, target = :out, args = (m[:y]::Real,), args_check = (args) -> args.m[:y] >= 0, body = (args) -> sqrt(args.m[:y]))
    # A check returning a string when it fails, a lazy one.
    @define_message_update_rule(
        node = Node, target = :y, args = (m[:p]::Real,),
        args_check = (args) -> 0 <= args.m[:p] <= 1 || lazy"a probability in [0, 1]; got $(args.m[:p])",
        body = (args) -> args.m[:p],
    )
    # A check over the algorithm, and one towards a group member, reading its index `k`.
    @define_message_update_rule(
        node = Node, target = :p, algorithm = Scaled, args = (m[:y]::Real,),
        args_check = (algo, args) -> args.m[:y] <= algo.limit, body = (args) -> args.m[:y],
    )
    @define_message_update_rule(
        node = Node, target = (:in, k), args = (m[:out]::Real,),
        args_check = (args) -> k == 1 || lazy"only the first member; got member $k", body = (args) -> args.m[:out],
    )
    # A check that returns neither a Bool nor a string.
    @define_message_update_rule(node = Node, target = :p, args = (m[:out]::Real,), args_check = (args) -> 1, body = (args) -> args.m[:out])
    # A marginal rule and an average energy, checked the same way.
    @define_marginal_update_rule(node = Node, target = (:out, :y), args = (m[:out]::Real, m[:y]::Real), args_check = (args) -> args.m[:y] != 0, body = (args) -> (args.m[:out], args.m[:y]))
    @define_average_energy(node = Node, args = (q[:out]::Real, q[:y]::Real), args_check = (args) -> args.q[:y] > 0, body = (args) -> log(args.q[:y]))
end

@testitem "args_check:a rule refuses inputs that fail its check" tags = [:base] setup = [CheckedRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: RuleInputError, list_rules
    C = CheckedRules
    failure(f) = (
        err = try
            f(); nothing
        catch e
            e
        end; err
    )

    @test getresult(call_message_update_rule(C.Node, :out; m = (y = 4.0,))) == 2.0
    err = failure(() -> call_message_update_rule(C.Node, :out; m = (y = -1.0,)))
    @test err isa RuleInputError && err.reason === nothing
    message = sprint(showerror, err)
    @test contains(message, "the message rule for Node towards :out under MessagePassingRulesBase.DefaultAlgorithm refuses its inputs: `args.m[:y] >= 0` is false")
    @test contains(message, "inputs: m[:y]::Float64") && contains(message, "rule at ")

    # A string is the reason, and always a failure.
    @test getresult(call_message_update_rule(C.Node, :y; m = (p = 0.3,))) == 0.3
    err = failure(() -> call_message_update_rule(C.Node, :y; m = (p = 1.5,)))
    @test err isa RuleInputError && err.reason == "a probability in [0, 1]; got 1.5"
    @test contains(sprint(showerror, err), "refuses its inputs: a probability in [0, 1]; got 1.5")

    # The check reads the algorithm and a group member's index as the body does.
    @test getresult(call_message_update_rule(C.Node, :p; m = (y = 1.0,), algorithm = C.Scaled(2.0))) == 1.0
    @test failure(() -> call_message_update_rule(C.Node, :p; m = (y = 3.0,), algorithm = C.Scaled(2.0))) isa RuleInputError
    @test getresult(call_message_update_rule(C.Node, (:in, 1); m = (out = 5.0,))) == 5.0
    @test failure(() -> call_message_update_rule(C.Node, (:in, 2); m = (out = 5.0,))).reason == "only the first member; got member 2"

    # A check must return a Bool or a string.
    err = failure(() -> call_message_update_rule(C.Node, :p; m = (out = 1.0,)))
    @test err isa ArgumentError && contains(err.msg, "returned a Int64") && contains(err.msg, "`false` or a string")

    # Marginal rules and average energies check their inputs too.
    @test getresult(call_marginal_update_rule(C.Node, (:out, :y); m = (out = 1.0, y = 2.0))) == (1.0, 2.0)
    @test failure(() -> call_marginal_update_rule(C.Node, (:out, :y); m = (out = 1.0, y = 0.0))) isa RuleInputError
    @test getresult(call_average_energy(C.Node; q = (out = 1.0, y = 1.0))) == 0.0
    @test contains(sprint(showerror, failure(() -> call_average_energy(C.Node; q = (out = 1.0, y = -1.0)))), "average energy for Node under MessagePassingRulesBase.DefaultAlgorithm refuses its inputs: `args.q[:y] > 0` is false")

    # The spec keeps the check's source, and shows it.
    spec = only(list_rules(C.Node, :out))
    @test spec.args_check == "args.m[:y] >= 0"
    @test contains(sprint(show, MIME"text/plain"(), spec), "checks:   args.m[:y] >= 0")
    @test contains(sprint(show, MIME"text/html"(), spec), "<th>checks</th>")
    @test only(list_rules(C.Node, :y)).args_check !== nothing
end

@testitem "args_check:a malformed check is refused at definition" tags = [:base] begin
    using MessagePassingRulesBase
    function failure(ex)
        try
            Core.eval(Module(), Expr(:toplevel, :(using MessagePassingRulesBase), ex)); ""
        catch e
            sprint(showerror, e isa LoadError ? e.error : e)
        end
    end
    node = :(struct N end; @define_factor_node(node = N, type = Stochastic, interfaces = [:out, :y]))
    rule(check) = Expr(:block, node, :(@define_message_update_rule(node = N, target = :out, args = (m[:y]::Real,), args_check = $check, body = (args) -> 1.0)))
    @test contains(failure(rule(:(isfinite))), "`args_check` is a function of the inputs")
    @test contains(failure(rule(:((out) -> true))), "unknown args_check slot `out`")
    @test contains(failure(rule(:((args, algo) -> true))), "canonical order")
end

@testitem "args_check:through message_passing_rule and the tables" tags = [:base] setup = [CheckedRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: message_passing_rule, Target, DefaultAlgorithm, RuleArgs, RuleInputError
    C = CheckedRules
    @test getresult(message_passing_rule(C.Node, Target(:out), DefaultAlgorithm(), RuleArgs(m = (y = 9.0,)))) == 3.0
    @test_throws RuleInputError message_passing_rule(C.Node, Target(:out), DefaultAlgorithm(), RuleArgs(m = (y = -9.0,)))
end

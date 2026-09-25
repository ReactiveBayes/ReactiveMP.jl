@testmodule LogScaleRules begin
    using MessagePassingRulesBase

    struct Scale end
    @define_factor_node(node = Scale, type = Deterministic, interfaces = [:out, :in, :w...])

    # A constant, kept as given: an `Irrational` promotes to whatever it is added to.
    @define_message_update_rule(node = Scale, target = :out, args = (m[:in]::Float32,), logscale = MathConstants.γ, body = (args) -> 2 * args.m[:in])
    @define_message_update_rule(node = Scale, target = :out, args = (m[:in]::Float64,), logscale = 0, body = (args) -> 2 * args.m[:in])
    # A function of the inputs.
    @define_message_update_rule(node = Scale, target = :in, args = (m[:out]::Real,), logscale = (args) -> -log(2.0), body = (args) -> args.m[:out] / 2)
    # From the body, reading the incoming log scales; the target's index is bound in both.
    @define_message_update_rule(
        node = Scale, target = (:w, k), args = (m[:out]::Real, m[:w][!k]::Real),
        logscale = from_body, reads_logscale = true,
        body = (args) -> with_logscale(k * args.m[:out], args.logscale.m[:out] + sum(something(l, 0.0) for l in args.logscale.m[:w])),
    )
    # None declared.
    @define_message_update_rule(node = Scale, target = :out, args = (m[:in]::Int,), body = (args) -> 2 * args.m[:in])
end

@testitem "logscale:declarations" tags = [:base] setup = [LogScaleRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: execute_rule, execute_rule_with_logscale, find_message_rule, RuleArgs, Target, IndexedTarget, RuleContext, NoAnnotations
    L = LogScaleRules

    constant = call_message_update_rule(L.Scale, :out; m = (in = 1.0f0,))
    @test getresult(constant) === 2.0f0
    @test getlogscale(constant) === MathConstants.γ
    @test typeof(getresult(constant) + getlogscale(constant)) === Float32
    @test getlogscale(call_message_update_rule(L.Scale, :out; m = (in = 1.0,))) === 0

    @test getlogscale(call_message_update_rule(L.Scale, :in; m = (out = 4.0,))) == -log(2.0)

    body = call_message_update_rule(L.Scale, (:w, 2); m = (out = 3.0, w = (1.0, nothing, 5.0)), logscale = (out = 0.5, w = (1.0, nothing, 2.0)))
    @test getresult(body) == 6.0
    @test getlogscale(body) == 3.5

    none = call_message_update_rule(L.Scale, :out; m = (in = 1,))
    @test getresult(none) === 2
    @test getlogscale(none) isa UndefinedLogScale
    @test getlogscale(none).cause === :no_declaration
    @test getlogscale(none).detail === getrule(none)

    # The internal paths: the result alone, or the result with its log scale.
    spec = find_message_rule(L.Scale, IndexedTarget(:w, 1), DefaultAlgorithm(), RuleArgs(m = (out = 1.0, w = (nothing, 2.0))))
    args = RuleArgs(m = (out = 1.0, w = (nothing, 2.0)), logscale = (out = 1.0, w = (nothing, 1.0)))
    @test execute_rule(spec, nothing, DefaultAlgorithm(), RuleContext(), args, NoAnnotations(), IndexedTarget(:w, 1)) == 1.0
    @test execute_rule_with_logscale(spec, nothing, nothing, DefaultAlgorithm(), RuleContext(), args, NoAnnotations(), IndexedTarget(:w, 1)) == (1.0, 2.0)
    @test spec.reads_logscale && spec.logscale === from_body
end

@testitem "logscale:reading incoming ones" tags = [:base] setup = [LogScaleRules] begin
    using MessagePassingRulesBase
    L = LogScaleRules

    # A rule that reads incoming log scales errors when none are given, naming itself.
    err = try
        call_message_update_rule(L.Scale, (:w, 2); m = (out = 3.0, w = (1.0, nothing)))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test contains(sprint(showerror, err), "reads the log scales of its inbound messages")
    @test contains(sprint(showerror, err), "Scale towards (:w, k)")
end

@testitem "logscale:undefined" tags = [:base] begin
    using MessagePassingRulesBase

    undefined = UndefinedLogScale(:initial)
    @test undefined + 1.0 === undefined
    @test 1.0 + undefined === undefined
    @test undefined + UndefinedLogScale(:fallback) === undefined
    @test +undefined === undefined
    @test !isdefined_logscale(undefined) && isdefined_logscale(0) && !isdefined_logscale(nothing)
    @test require_logscale(-1.5) === -1.5
    @test_throws UndefinedLogScaleError require_logscale(undefined)
    @test_throws ArgumentError require_logscale(nothing)
    @test contains(sprint(showerror, UndefinedLogScaleError(undefined)), "an initial one")
    @test contains(sprint(showerror, UndefinedLogScaleError(UndefinedLogScale(:no_compute_logscale, (Int, Float64)))), "Int64 and Float64")
    @test repr(undefined) == "UndefinedLogScale(:initial)"
    @test contains(repr(MIME"text/plain"(), UndefinedLogScale(:form_constraint)), "form constraint")
    @test with_logscale(1, 2) === with_logscale(result = 1, logscale = 2)
end

@testitem "logscale:display" tags = [:base] setup = [LogScaleRules] begin
    using MessagePassingRulesBase
    L = LogScaleRules

    body = call_message_update_rule(L.Scale, (:w, 2); m = (out = 3.0, w = (1.0, nothing)), logscale = (out = 0.5, w = (1.0, nothing)))
    @test repr(body) == "RuleResult(6.0, logscale = 1.5)"
    text = repr(MIME"text/plain"(), body)
    @test contains(text, "message of Scale towards (:w, 2)")
    @test contains(text, "logscale   1.5  (computed by the body)")
    @test contains(text, "out   ──▶  m  3.0")
    @test contains(text, "incoming   m[:out] = 0.5")
    @test contains(repr(MIME"text/plain"(), call_message_update_rule(L.Scale, :out; m = (in = 1,))), "declares no `logscale`")
    spec = getrule(body)
    @test contains(repr(MIME"text/plain"(), spec), "logscale: from the body · reads incoming log scales")
end

@testitem "logscale:malformed" tags = [:base] begin
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
    rule(kw...) = :(
        struct N end;
        @define_factor_node(node = N, type = Stochastic, interfaces = [:out, :in]);
        @define_message_update_rule(node = N, target = :out, args = (m[:in]::Any,), $(kw...));
        call_message_update_rule(N, :out; m = (in = 1.0,))
    )

    @test failure(rule(:(logscale = "zero"), :(body = (args) -> 1))) |> msg -> contains(msg, "a number, a function of its inputs or `from_body`")
    @test failure(rule(:(reads_logscale = 1), :(body = (args) -> 1))) |> msg -> contains(msg, "`reads_logscale` must be `true` or `false`")
    @test failure(rule(:(logscale = (ann) -> 0), :(body = (args) -> 1))) |> msg -> contains(msg, "unknown logscale slot")
    @test failure(rule(:(logscale = from_body), :(body = (args) -> 1))) |> msg -> contains(msg, "did not return `with_logscale(result, logscale)`")
    @test failure(rule(:(body = (args) -> with_logscale(1, 0)))) |> msg -> contains(msg, "does not declare `logscale = from_body`")
    @test failure(
        :(
            struct N end;
            @define_factor_node(node = N, type = Stochastic, interfaces = [:out, :in]);
            @define_marginal_update_rule(node = N, target = (:out, :in), args = (m[:in]::Any,), logscale = 0, body = (args) -> 1)
        )
    ) |> msg -> contains(msg, "logscale")
    @test failure(
        :(
            struct N end;
            @define_factor_node(node = N, type = Stochastic, interfaces = [:out, :in]);
            @define_average_energy(node = N, args = (q[:in]::Any,), reads_logscale = true, body = (args) -> 1)
        )
    ) |> msg -> contains(msg, "reads_logscale")
end

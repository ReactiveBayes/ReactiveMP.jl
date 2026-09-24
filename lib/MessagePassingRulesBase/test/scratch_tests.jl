@testmodule ScratchNodes begin
    using MessagePassingRulesBase

    struct Summing end
    @define_factor_node(node = Summing, type = Stochastic, interfaces = [:out, :in, :w...])

    # How many scratches have been built, to tell a reused one from a fresh one.
    const BUILT = Ref(0)

    # The sum of `in` weighted elementwise, through a work vector the scratch holds.
    @define_message_update_rule(
        node = Summing, target = :out, args = (m[:in]::Vector{Float64},),
        scratch = (args) -> (BUILT[] += 1; (work = similar(args.m[:in]),)),
        body = (scratch, args) -> begin
            scratch.work .= 2 .* args.m[:in]
            sum(scratch.work)
        end,
    )

    # In place and with scratch: `output` first, then `scratch`.
    @define_message_update_rule(
        node = Summing, target = :in, args = (m[:out]::Vector{Float64},),
        inplace = true, preallocate = (args) -> similar(args.m[:out]),
        scratch = (args) -> (work = similar(args.m[:out]),),
        body = (output, scratch, args) -> begin
            scratch.work .= args.m[:out] .+ 1
            output .= scratch.work .* 3
        end,
    )

    # A group target: the scratch builder, like `preallocate`, binds the target's index `k`.
    @define_message_update_rule(
        node = Summing, target = (:w, k), args = (m[:in]::Vector{Float64},),
        scratch = (args) -> (work = fill(Float64(k), length(args.m[:in])),),
        body = (scratch, args) -> begin
            scratch.work .= k .* args.m[:in]
            sum(scratch.work)
        end,
    )

    @define_marginal_update_rule(
        node = Summing, target = (:out, :in), args = (m[:out]::Vector{Float64}, m[:in]::Vector{Float64}),
        scratch = (args) -> (work = similar(args.m[:out]),),
        body = (scratch, args) -> (scratch.work .= args.m[:out] .- args.m[:in]; sum(scratch.work)),
    )

    # An average energy has no scratch.
    @define_average_energy(node = Summing, args = (q[:in]::Vector{Float64},), body = (args) -> sum(args.q[:in]))
end

@testitem "scratch:declared and run" tags = [:base] setup = [ScratchNodes] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: find_message_rule, find_marginal_rule, execute_rule, rule_scratch, rule_algorithm,
        Target, IndexedTarget, ClusterTarget, DefaultAlgorithm, RuleContext, NoAnnotations, RuleArgs
    S = ScratchNodes

    args = RuleArgs(m = (in = [1.0, 2.0, 3.0],))
    spec = find_message_rule(S.Summing, Target(:out), DefaultAlgorithm(), args)
    @test spec.scratch !== nothing && !spec.inplace && spec.pure

    # The allocating path builds a fresh scratch on every call.
    S.BUILT[] = 0
    @test message_passing_rule(S.Summing, Target(:out), DefaultAlgorithm(), args) == 12.0
    @test message_passing_rule(S.Summing, Target(:out), DefaultAlgorithm(), args) == 12.0
    @test S.BUILT[] == 2

    # An engine builds one with `rule_scratch` and passes it on every call.
    S.BUILT[] = 0
    scratch = rule_scratch(spec, DefaultAlgorithm(), RuleContext(), args, Target(:out))
    for _ in 1:3
        @test execute_rule(spec, nothing, scratch, DefaultAlgorithm(), RuleContext(), args, NoAnnotations(), Target(:out)) == 12.0
    end
    @test S.BUILT[] == 1
    @test scratch.work == [2.0, 4.0, 6.0]
    # Reusing it saves the scratch's own allocation: the work vector, three Float64s.
    fresh(spec, args) = @allocated execute_rule(spec, nothing, DefaultAlgorithm(), RuleContext(), args, NoAnnotations(), Target(:out))
    reused(spec, scratch, args) = @allocated execute_rule(spec, nothing, scratch, DefaultAlgorithm(), RuleContext(), args, NoAnnotations(), Target(:out))
    fresh(spec, args), reused(spec, scratch, args)
    @test fresh(spec, args) - reused(spec, scratch, args) >= 3 * sizeof(Float64)

    # A rule without scratch has none to build.
    energy_args = RuleArgs(q = (in = [1.0, 2.0],))
    energy = MessagePassingRulesBase.find_average_energy(S.Summing, DefaultAlgorithm(), energy_args)
    @test energy.scratch === nothing
    @test rule_scratch(energy, DefaultAlgorithm(), RuleContext(), energy_args, nothing) === nothing

    # In place with scratch.
    inargs = RuleArgs(m = (out = [1.0, 2.0],))
    inspec = find_message_rule(S.Summing, Target(:in), DefaultAlgorithm(), inargs)
    output = similar([1.0, 2.0])
    @test message_passing_rule!(output, S.Summing, Target(:in), DefaultAlgorithm(), inargs) === output
    @test output == [6.0, 9.0]

    # A group target's scratch is built with its index.
    wspec = find_message_rule(S.Summing, IndexedTarget(:w, 3), DefaultAlgorithm(), args)
    @test rule_scratch(wspec, DefaultAlgorithm(), RuleContext(), args, IndexedTarget(:w, 3)).work == [3.0, 3.0, 3.0]
    @test message_passing_rule(S.Summing, IndexedTarget(:w, 2), DefaultAlgorithm(), args) == 12.0

    # A marginal rule takes scratch too.
    margs = RuleArgs(m = (out = [3.0, 1.0], in = [1.0, 1.0]))
    @test message_passing_marginalrule(S.Summing, ClusterTarget((:out, :in)), DefaultAlgorithm(), margs) == 2.0

    # The display says so.
    @test contains(sprint(show, MIME("text/plain"), spec), "scratch: yes")
end

@testitem "scratch:malformed" tags = [:base] begin
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
        @define_message_update_rule(node = N, target = :out, args = (m[:in]::Any,), $(kw...))
    )

    @test failure(rule(:(body = (scratch, args) -> 1))) |> msg -> contains(msg, "the `scratch` slot requires a `scratch` keyword")
    @test failure(rule(:(scratch = (args) -> 1), :(body = (args) -> 1))) |> msg -> contains(msg, "declares `scratch` but its body does not take it")
    @test failure(rule(:(inplace = true), :(preallocate = (args) -> [0.0]), :(scratch = (args) -> 1), :(body = (scratch, output, args) -> 1))) |>
        msg -> contains(msg, "canonical order")
    @test failure(
        :(
            struct N end;
            @define_factor_node(node = N, type = Stochastic, interfaces = [:out, :in]);
            @define_average_energy(node = N, args = (q[:in]::Any,), scratch = (args) -> 1, body = (scratch, args) -> 1)
        )
    ) |> msg -> contains(msg, "scratch")
end


@testitem "rule" begin
    using ReactiveMP, MacroTools, Logging, BayesBase, Distributions, ExponentialFamily

    import MacroTools: inexpr

    @testset "Testing utilities" begin
        import ReactiveMP: TestRulesConfiguration
        import ReactiveMP: check_type_promotion, check_type_promotion!
        import ReactiveMP: float_tolerance, float_tolerance!
        import ReactiveMP: extra_float_types, extra_float_types!
        import ReactiveMP: test_rules_parse_configuration

        @testset "TestRulesConfiguration" begin

            # check_type_promotion setter test
            let configuration = TestRulesConfiguration()
                for check in (true, false)
                    check_type_promotion!(configuration, check)
                    @test check_type_promotion(configuration) === check
                end
            end

            # check atol can be set as a single number
            let configuration = TestRulesConfiguration()
                for atol in (1e-6, 1e-12)
                    float_tolerance!(configuration, atol)
                    @test all(tolerance -> isequal(tolerance, atol), values(float_tolerance(configuration)))
                end
            end

            # check atol can be set as an array of pairs
            let configuration = TestRulesConfiguration()
                for atol in [[Float32 => 1e-5, Float64 => 1e-11], [Float32 => 1e-4, Float64 => 1e-10]]
                    float_tolerance!(configuration, atol)
                    for (key, value) in atol
                        @test isequal(float_tolerance(configuration, key), value)
                    end
                end
            end

            # check atol can be set individually
            let configuration = TestRulesConfiguration()
                for T in [Float32, Float16, BigFloat, Int], atol in (1e-6, 1e-12)
                    float_tolerance!(configuration, T, atol)
                    @test isequal(float_tolerance(configuration, T), atol)
                end
            end

            # extra_float_types setter test
            let configuration = TestRulesConfiguration()
                for extra_types in ([Float64, Float32], [Int, BigFloat])
                    extra_float_types!(configuration, extra_types)
                    @test isequal(extra_float_types(configuration), extra_types)
                end
            end

            @test_throws ErrorException test_rules_parse_configuration(:configuration, :([1]))
            @test_throws ErrorException test_rules_parse_configuration(:configuration, :([options = value = broken]))

            for name in (:configuration, :blabla),
                check in (true, false),
                atol in (1e-4, :([Float64 => 1e-11, Float32 => 1e-4])),
                extra_types in (:([Float64]), :([Float32, BigFloat]))

                expression = test_rules_parse_configuration(name, :([check_type_promotion = $check, atol = $atol, extra_float_types = $extra_types]))

                @test inexpr(expression, :(ReactiveMP.check_type_promotion!($name, convert(Bool, $check))))
                @test inexpr(expression, :(ReactiveMP.float_tolerance!($name, $atol)))
                @test inexpr(expression, :(ReactiveMP.extra_float_types!($name, $extra_types)))
            end
        end

        @testset "test_rules_generate_testset" begin
            import ReactiveMP: test_rules_generate_testset, TestRuleEntry, TestRuleEntryInputSpecification
            import ReactiveMP: CallRuleMacroFnExpr, CallMarginalRuleMacroFnExpr

            fns = (CallRuleMacroFnExpr, CallMarginalRuleMacroFnExpr)
            tfns = (:f, :test)
            specs = (:(NormalMeanVariance(:out, Marginalisation)), :(Gamma(:out, MomentMatching)))
            inputs = (:((m_mean = NormalMeanVariance(0.0, 1.0), q_var = InverseGamma(1.0, 1.0))),)
            outputs = (:(NormalMeanVariance(0.0, 0.0)), :(Gamma(2.0, 3.0)))

            for f in fns, test_f in tfns, spec in specs, input in inputs, output in outputs
                test_entry = convert(TestRuleEntry, :((input = $input, output = $output)))
                expression = test_rules_generate_testset(test_entry, test_f, f, spec, :configuration)
                @test inexpr(expression, output)
                @test inexpr(expression, test_f)
                @test inexpr(expression, :(ReactiveMP.float_tolerance))
                @test inexpr(expression, :(ReactiveMP.custom_rule_isapprox))
                @test inexpr(expression, :(ReactiveMP.BayesBase.isequal_typeof))
            end
        end

        @testset "TestRuleEntryInputSpecification" begin
            import ReactiveMP: TestRuleEntryInputSpecification

            let spec = TestRuleEntryInputSpecification([:m_x => 1], nothing)
                @test convert(Expr, spec) == :((m_x = 1,))
            end

            let spec = TestRuleEntryInputSpecification([:q_x => 1, :m_y => :(Normal(1.0, 1.0))], :(Meta(2)))
                @test convert(Expr, spec) == :((q_x = 1, m_y = Normal(1.0, 1.0), meta = Meta(2)))
            end
        end

        @testset "convert(TestRuleEntryInputSpecification, expression)" begin
            import ReactiveMP: TestRuleEntryInputSpecification

            @test convert(TestRuleEntryInputSpecification, :((key1 = 1, key2 = 3, key3 = 2))) == TestRuleEntryInputSpecification([:key1 => 1, :key2 => 3, :key3 => 2], nothing)
            @test convert(TestRuleEntryInputSpecification, :((key1 = 1, key2 = 2, key3 = 3))) == TestRuleEntryInputSpecification([:key1 => 1, :key2 => 2, :key3 => 3], nothing)
            @test convert(TestRuleEntryInputSpecification, :((key1 = Gamma(1.0, 1.0), key2 = NormalMeanVariance(0.0, 1.0), key3 = 3))) ==
                TestRuleEntryInputSpecification([:key1 => :(Gamma(1.0, 1.0)), :key2 => :(NormalMeanVariance(0.0, 1.0)), :key3 => 3], nothing)
            @test convert(TestRuleEntryInputSpecification, :((key1 = 1, key2 = 2, key3 = 3, meta = "hello"))) ==
                TestRuleEntryInputSpecification([:key1 => 1, :key2 => 2, :key3 => 3], :("hello"))
            @test convert(TestRuleEntryInputSpecification, :((key1 = 1, key2 = ManyOf(1, 2), key3 = 3, meta = "hello"))) ==
                TestRuleEntryInputSpecification([:key1 => 1, :key2 => :(ManyOf(1, 2)), :key3 => 3], :("hello"))
        end

        @testset "TestRuleEntry" begin
            import ReactiveMP: TestRuleEntry

            let spec = TestRuleEntry(TestRuleEntryInputSpecification([:m_x => 1], nothing), :(Normal(0.0, 3.0)))
                @test convert(Expr, spec) == :((input = (m_x = 1,), output = Normal(0.0, 3.0)))
            end

            let spec = TestRuleEntry(TestRuleEntryInputSpecification([:q_x => 1, :m_y => :(Normal(1.0, 1.0))], :(Meta(2))), :(Gamma(2.0, 3.0)))
                @test convert(Expr, spec) == :((input = (q_x = 1, m_y = Normal(1.0, 1.0), meta = Meta(2)), output = Gamma(2.0, 3.0)))
            end
        end

        @testset "convert(TestRuleEntry, expression)" begin
            import ReactiveMP: TestRuleEntry

            @test_throws MethodError convert(TestRuleEntry, :(1))
            @test_throws ErrorException convert(TestRuleEntry, :([1]))
            @test_throws ErrorException convert(TestRuleEntry, :([input, output]))
            @test_throws ErrorException convert(TestRuleEntry, :([input = 2, output]))
            @test_throws ErrorException convert(TestRuleEntry, :([input = 2, output]))

            let entry = convert(TestRuleEntry, :(input = (m_x = Normal(3.0, 3.0),), output = 3))
                @test length(entry.input.arguments) === 1
                @test inexpr(entry.input.arguments[1][1], :m_x)
                @test inexpr(entry.input.arguments[1][2], :(Normal(3.0, 3.0)))
                @test inexpr(entry.output, :(3))
            end

            let entries = convert(
                    Vector{TestRuleEntry},
                    :([
                        (input = (m_x = Normal(1.0, 2.0), q_y = PointMass(3)), output = Gamma(1.0, 2.0)),
                        (input = (q_x = Normal(2.0, 3.0), m_y = PointMass(4), meta = Meta("hello")), output = Gamma(2.0, 3.0))
                    ])
                )
                @test length(entries) === 2

                @test length(entries[1].input.arguments) === 2
                @test inexpr(entries[1].input.arguments[1][1], :m_x)
                @test inexpr(entries[1].input.arguments[1][2], :(Normal(1.0, 2.0)))
                @test inexpr(entries[1].input.arguments[2][1], :q_y)
                @test inexpr(entries[1].input.arguments[2][2], :(PointMass(3)))
                @test isnothing(entries[1].input.meta)
                @test inexpr(entries[1].output, :(Gamma(1.0, 2.0)))

                @test length(entries[2].input.arguments) === 2
                @test inexpr(entries[2].input.arguments[1][1], :q_x)
                @test inexpr(entries[2].input.arguments[1][2], :(Normal(2.0, 3.0)))
                @test inexpr(entries[2].input.arguments[2][1], :m_y)
                @test inexpr(entries[2].input.arguments[2][2], :(PointMass(4)))
                @test inexpr(entries[2].input.meta, :(Meta("hello")))
                @test inexpr(entries[2].output, :(Gamma(2.0, 3.0)))
            end
        end

        @testset "test_rules_convert_paramfloattype_for_test_entry" begin
            import ReactiveMP: TestRuleEntry, TestRuleEntryInputSpecification, test_rules_convert_paramfloattype_for_test_entry

            for m in (1, :(Normal(0.0, 1.0))), v in (2, Gamma(2.0, 3.0)), output in (3, :(Normal(2.0, 3.0))), eltype in (:Float32, Float64)
                let test_entry = TestRuleEntry(TestRuleEntryInputSpecification([:m => m, :v => v], :(Meta(1))), output)
                    modified_inputs = map(e -> convert(Expr, e), test_rules_convert_paramfloattype_for_test_entry(test_entry, eltype))

                    modified_m = :(ReactiveMP.BayesBase.convert_paramfloattype($eltype, $m))
                    modified_v = :(ReactiveMP.BayesBase.convert_paramfloattype($eltype, $v))
                    original_meta = :(meta = Meta(1))
                    modified_meta = :(ReactiveMP.BayesBase.convert_paramfloattype($eltype, Meta(1)))

                    @test all(modified_inputs) do expression
                        !inexpr(expression, modified_meta)
                    end

                    @test any(modified_inputs) do expression
                        inexpr(expression, original_meta)
                    end

                    @test any(modified_inputs) do expression
                        inexpr(expression, modified_m)
                    end

                    @test any(modified_inputs) do expression
                        inexpr(expression, modified_v)
                    end

                    @test any(modified_inputs) do expression
                        inexpr(expression, modified_m)
                        inexpr(expression, modified_v)
                    end
                end
            end
        end

        @testset "test_rules_convert_paramfloattype" begin
            import ReactiveMP: test_rules_convert_paramfloattype

            for eltype in (:Float32, :Float64)
                @test inexpr(
                    test_rules_convert_paramfloattype(:(NormalMeanVariance(1.0, 2.0)), eltype),
                    :(ReactiveMP.BayesBase.convert_paramfloattype($eltype, NormalMeanVariance(1.0, 2.0)))
                )
                @test inexpr(
                    test_rules_convert_paramfloattype(:(m_in = NormalMeanVariance(1.0, 2.0)), eltype),
                    :(m_in = ReactiveMP.BayesBase.convert_paramfloattype($eltype, NormalMeanVariance(1.0, 2.0)))
                )
                @test inexpr(
                    test_rules_convert_paramfloattype(:((m_in = NormalMeanVariance(1.0, 2.0),)), eltype),
                    :((m_in = ReactiveMP.BayesBase.convert_paramfloattype($eltype, NormalMeanVariance(1.0, 2.0)),))
                )
                @test inexpr(
                    test_rules_convert_paramfloattype(:((m_in = ManyOf(NormalMeanVariance(1.0, 2.0)),)), eltype),
                    :((m_in = ManyOf(ReactiveMP.BayesBase.convert_paramfloattype($eltype, NormalMeanVariance(1.0, 2.0)))))
                )
                @test inexpr(
                    test_rules_convert_paramfloattype(:((m_in = NormalMeanVariance(1.0, 2.0), q_out = Gamma(1.0, 2.0))), eltype),
                    :((
                        m_in = ReactiveMP.BayesBase.convert_paramfloattype($eltype, NormalMeanVariance(1.0, 2.0)),
                        q_out = ReactiveMP.BayesBase.convert_paramfloattype($eltype, Gamma(1.0, 2.0))
                    ))
                )
            end
        end

        @testset "`@rule` invalid input arguments" begin
            struct MyNode end

            @eval @rule MyNode(:out, Marginalisation) (m_a::PointMass, q_b::PointMass, meta::Int) = begin end
            @test_throws LoadError @eval @rule MyNode(:out, Marginalisation) (a::PointMass, b::PointMass) = begin end
        end

        @testset "`@marginalrule` invalid input arguments" begin
            struct MyNode end

            @eval @marginalrule MyNode(:a_b) (m_a::PointMass, m_b::PointMass, q_c::PointMass, meta::Int) = begin end
            @test_throws LoadError @eval @marginalrule MyNode(:out) (a::PointMass, b::PointMass, c::PointMass, meta::Int) = begin end
        end

        @testset "basic `test_rule` macro invokation" begin
            struct TestNodeForTestRuleMacro end

            @node TestNodeForTestRuleMacro Stochastic [out, x, y]

            @rule TestNodeForTestRuleMacro(:out, Marginalisation) (m_x::PointMass, q_y::PointMass) = PointMass(mean(m_x) + mean(q_y))
            @rule TestNodeForTestRuleMacro(:out, Marginalisation) (m_x::ManyOf{N, Any}, q_y::PointMass) where {N} = PointMass(sum(mean, m_x) + mean(q_y))
            @rule TestNodeForTestRuleMacro(:out, Marginalisation) (m_x::PointMass, q_y::PointMass, meta::typeof(-)) = PointMass(mean(m_x) - mean(q_y))

            ReactiveMP.@test_rules [check_type_promotion = true] TestNodeForTestRuleMacro(:out, Marginalisation) [
                (input = (m_x = PointMass(1), q_y = PointMass(2)), output = PointMass(3)),
                (input = (m_x = PointMass(1), q_y = PointMass(2)), output = PointMass(3)),
                (input = (m_x = ManyOf(PointMass(1), PointMass(1)), q_y = PointMass(2)), output = PointMass(4)),
                (input = (m_x = ManyOf(PointMass(1), PointMass(2)), q_y = PointMass(2)), output = PointMass(5)),
                (input = (m_x = PointMass(1), q_y = PointMass(2), meta = -), output = PointMass(-1)),
                (input = (m_x = PointMass(3), q_y = PointMass(2), meta = -), output = PointMass(1))
            ]

            struct TestMetaForFailingRule end

            # This rule violates type_promotion and must fail
            @rule TestNodeForTestRuleMacro(:out, Marginalisation) (m_x::PointMass, q_y::PointMass, meta::TestMetaForFailingRule) = PointMass(1.0)

            ReactiveMP.@test_rules [check_type_promotion = false] TestNodeForTestRuleMacro(:out, Marginalisation) [(
                input = (m_x = PointMass(1), q_y = PointMass(2), meta = TestMetaForFailingRule()), output = PointMass(1.0)
            )]

            io = IOBuffer()
            logger = SimpleLogger(io)

            tests_status = []

            with_logger(logger) do
                ReactiveMP.@test_rules (status) -> push!(tests_status, status) [check_type_promotion = true] TestNodeForTestRuleMacro(:out, Marginalisation) [(
                    input = (m_x = PointMass(1), q_y = PointMass(2), meta = TestMetaForFailingRule()), output = PointMass(1.0)
                )]
            end

            # Here assume that `2:end` tests are promotion tests
            @test tests_status[1] === true
            @test tests_status[end] === false # Assume that the last test is the `BigFloat` promotion test
            @test sum(tests_status[2:end]) < length(tests_status[2:end])

            test_output = String(take!(io))

            @test occursin("Testset for rule TestNodeForTestRuleMacro(:out, Marginalisation) has failed", test_output)
        end
    end

    @testset "Macro utilities" begin
        import ReactiveMP: rule_macro_parse_on_tag, rule_macro_parse_fn_args, call_rule_macro_parse_fn_args, rule_macro_check_fn_args

        @testset "rule_macro_parse_on_tag(expression)" begin
            @test rule_macro_parse_on_tag(:(:out)) == (:(Val{:out}), nothing, nothing)
            @test rule_macro_parse_on_tag(:(:mean)) == (:(Val{:mean}), nothing, nothing)
            @test rule_macro_parse_on_tag(:(:mean, k)) == (:(Tuple{Val{:mean}, Int}), :k, :(k = on[2]))
            @test rule_macro_parse_on_tag(:(:precision, r)) == (:(Tuple{Val{:precision}, Int}), :r, :(r = on[2]))

            @test_throws ErrorException rule_macro_parse_on_tag(:(out))
            @test_throws ErrorException rule_macro_parse_on_tag(:(123))
            @test_throws ErrorException rule_macro_parse_on_tag(:(:mean, 1))
            @test_throws ErrorException rule_macro_parse_on_tag(:(precision, r))
        end

        @testset "rule_macro_check_fn_args(inputs; allowed_inputs, allowed_prefixes)" begin
            @test rule_macro_check_fn_args([(:m_a, 1), (:m_b, 2)]; allowed_inputs = (), allowed_prefixes = (:m_,))
            @test rule_macro_check_fn_args([(:q_a, 1), (:m_b, 2)]; allowed_inputs = (), allowed_prefixes = (:m_, :q_))
            @test rule_macro_check_fn_args([(:meta, 3)]; allowed_inputs = (:meta,), allowed_prefixes = (:m_, :q_))
            @test rule_macro_check_fn_args([(:q_a, 1), (:m_b, 2), (:meta, 3)]; allowed_inputs = (:meta,), allowed_prefixes = (:m_, :q_))

            @test_throws ErrorException rule_macro_check_fn_args([(:a, 1), (:b, 2)]; allowed_inputs = (), allowed_prefixes = (:m_, :q_))
            @test_throws ErrorException rule_macro_check_fn_args([(:meta, 3)]; allowed_inputs = (), allowed_prefixes = (:m_, :q_))
        end

        @testset "rule_macro_parse_fn_args(inputs; specname, prefix, proxy)" begin
            names, types, init = rule_macro_parse_fn_args(
                [(:m_out, :PointMass), (:m_mean, :NormalMeanPrecision)]; specname = :messages, prefix = :m_, proxy = :(ReactiveMP.Message)
            )

            @test names == :(Val{(:out, :mean)})
            @test types == :(Tuple{ReactiveMP.Message{<:PointMass}, ReactiveMP.Message{<:NormalMeanPrecision}})
            @test init == Expr[:(m_out = getdata(messages[1])), :(m_mean = getdata(messages[2]))]

            names, types, init = rule_macro_parse_fn_args(
                [(:m_out, :PointMass), (:m_mean, :NormalMeanPrecision)]; specname = :marginals, prefix = :q_, proxy = :(ReactiveMP.Marginal)
            )

            @test names == :Nothing
            @test types == :Nothing
            @test init == Expr[]

            names, types, init = rule_macro_parse_fn_args(
                [(:m_out, :PointMass), (:q_mean, :NormalMeanPrecision)]; specname = :marginals, prefix = :q_, proxy = :(ReactiveMP.Marginal)
            )

            @test names == :(Val{(:mean,)})
            @test types == :(Tuple{ReactiveMP.Marginal{<:NormalMeanPrecision}})
            @test init == Expr[:(q_mean = getdata(marginals[1]))]
        end

        @testset "call_rule_macro_parse_fn_args(inputs; specname, prefix, proxy)" begin
            names, values = call_rule_macro_parse_fn_args(
                [(:m_out, :(PointMass(1.0))), (:m_mean, :(NormalMeanPrecision(0.0, 1.0)))]; specname = :messages, prefix = :m_, proxy = :(ReactiveMP.Message)
            )

            @test names == :(Val{(:out, :mean)}())
            @test values == :(ReactiveMP.Message(PointMass(1.0), false, false, nothing), ReactiveMP.Message(NormalMeanPrecision(0.0, 1.0), false, false, nothing))

            names, values = call_rule_macro_parse_fn_args(
                [(:m_out, :(PointMass(1.0))), (:m_mean, :(NormalMeanPrecision(0.0, 1.0)))]; specname = :marginals, prefix = :q_, proxy = :(ReactiveMP.Marginal)
            )

            @test names == :nothing
            @test values == :nothing

            names, values = call_rule_macro_parse_fn_args(
                [(:m_out, :(PointMass(1.0))), (:q_mean, :(NormalMeanPrecision(0.0, 1.0)))]; specname = :marginals, prefix = :q_, proxy = :(ReactiveMP.Marginal)
            )

            @test names == :(Val{(:mean,)}())
            @test values == :((ReactiveMP.Marginal(NormalMeanPrecision(0.0, 1.0), false, false, nothing),))
        end
    end

    @testset "Error utilities" begin
        @testset "rule_method_error" begin
            as_vague_msg(::Type{T}) where {T} = Message(vague(T), false, false, nothing)
            as_vague_mrg(::Type{T}) where {T} = Marginal(vague(T), false, false, nothing)

            let
                err = ReactiveMP.RuleMethodError(
                    NormalMeanPrecision,
                    Val{:μ}(),
                    Marginalisation(),
                    Val{(:out,)}(),
                    (as_vague_msg(NormalMeanVariance),),
                    Val{(:τ,)}(),
                    (as_vague_mrg(Gamma),),
                    nothing,
                    nothing,
                    nothing
                )

                io = IOBuffer()
                showerror(io, err)
                output = String(take!(io))

                @test occursin("Possible fix, define:", output)
                @test occursin("@rule", output)
                @test occursin("Marginalisation", output)
                @test occursin("m_out::NormalMeanVariance", output)
                @test occursin("q_τ::Gamma", output)
            end

            let
                err = ReactiveMP.RuleMethodError(
                    NormalMeanPrecision,
                    Val{:μ}(),
                    Marginalisation(),
                    Val{(:out,)}(),
                    (as_vague_msg(NormalMeanVariance),),
                    Val{(:τ,)}(),
                    (as_vague_mrg(Gamma),),
                    1.0,
                    nothing,
                    nothing
                )

                io = IOBuffer()
                showerror(io, err)
                output = String(take!(io))

                @test occursin("Possible fix, define:", output)
                @test occursin("@rule", output)
                @test occursin("Marginalisation", output)
                @test occursin("m_out::NormalMeanVariance", output)
                @test occursin("q_τ::Gamma", output)
                @test occursin("meta::Float64", output)
            end

            let
                err = ReactiveMP.RuleMethodError(
                    NormalMeanPrecision,
                    Val{:μ}(),
                    Marginalisation(),
                    Val{(:out, :τ)}(),
                    (as_vague_msg(NormalMeanVariance), as_vague_msg(Gamma)),
                    nothing,
                    nothing,
                    1.0,
                    nothing,
                    nothing
                )

                io = IOBuffer()
                showerror(io, err)
                output = String(take!(io))

                @test occursin("Possible fix, define:", output)
                @test occursin("@rule", output)
                @test occursin("Marginalisation", output)
                @test occursin("m_out::NormalMeanVariance", output)
                @test occursin("m_τ::Gamma", output)
                @test occursin("meta::Float64", output)
            end

            let
                err = ReactiveMP.RuleMethodError(
                    NormalMeanPrecision,
                    Val{:μ}(),
                    Marginalisation(),
                    nothing,
                    nothing,
                    Val{(:out, :τ)}(),
                    (as_vague_mrg(NormalMeanVariance), as_vague_mrg(Gamma)),
                    1.0,
                    nothing,
                    nothing
                )

                io = IOBuffer()
                showerror(io, err)
                output = String(take!(io))

                @test occursin("Possible fix, define:", output)
                @test occursin("@rule", output)
                @test occursin("Marginalisation", output)
                @test occursin("q_out::NormalMeanVariance", output)
                @test occursin("q_τ::Gamma", output)
                @test occursin("meta::Float64", output)
            end

            let
                err = ReactiveMP.RuleMethodError(
                    NormalMeanPrecision,
                    Val{:τ}(),
                    Marginalisation(),
                    nothing,
                    nothing,
                    Val{(:out_μ,)}(),
                    (Marginal(vague(MvNormalMeanPrecision, 2), false, false, nothing),),
                    1.0,
                    nothing,
                    nothing
                )

                io = IOBuffer()
                showerror(io, err)
                output = String(take!(io))

                @test occursin("Possible fix, define:", output)
                @test occursin("@rule", output)
                @test occursin("Marginalisation", output)
                @test occursin("q_out_μ::MvNormalMeanPrecision", output)
                @test occursin("meta::Float64", output)
            end

            let
                err = ReactiveMP.RuleMethodError(
                    NormalMeanPrecision,
                    Val{:τ}(),
                    Marginalisation(),
                    Val{(:out, :μ)}(),
                    (as_vague_msg(NormalMeanVariance), as_vague_msg(NormalMeanVariance)),
                    Val{(:out_μ,)}(),
                    (Marginal(vague(MvNormalMeanPrecision, 2), false, false, nothing),),
                    1.0,
                    nothing,
                    nothing
                )

                io = IOBuffer()
                showerror(io, err)
                output = String(take!(io))

                @test occursin("Possible fix, define", output)
                @test occursin("(m_out::NormalMeanVariance, m_μ::NormalMeanVariance, q_out_μ::MvNormalMeanPrecision, meta::Float64)", output)
            end
        end

        @testset "marginalrule_method_error" begin
            as_vague_msg(::Type{T}) where {T} = Message(vague(T), false, false, nothing)
            as_vague_mrg(::Type{T}) where {T} = Marginal(vague(T), false, false, nothing)

            let
                err = ReactiveMP.MarginalRuleMethodError(
                    NormalMeanPrecision, Val{:μ}(), Val{(:out,)}(), (as_vague_msg(NormalMeanVariance),), Val{(:τ,)}(), (as_vague_mrg(Gamma),), nothing, nothing
                )

                io = IOBuffer()
                showerror(io, err)
                output = String(take!(io))

                @test occursin("Possible fix, define:", output)
                @test occursin("@marginalrule", output)
                @test occursin("m_out::NormalMeanVariance", output)
                @test occursin("q_τ::Gamma", output)
            end

            let
                err = ReactiveMP.MarginalRuleMethodError(
                    NormalMeanPrecision, Val{:μ}(), Val{(:out,)}(), (as_vague_msg(NormalMeanVariance),), Val{(:τ,)}(), (as_vague_mrg(Gamma),), 1.0, nothing
                )

                io = IOBuffer()
                showerror(io, err)
                output = String(take!(io))

                @test occursin("Possible fix, define:", output)
                @test occursin("@marginalrule", output)
                @test occursin("m_out::NormalMeanVariance", output)
                @test occursin("q_τ::Gamma", output)
                @test occursin("meta::Float64", output)
            end

            let
                err = ReactiveMP.MarginalRuleMethodError(
                    NormalMeanPrecision, Val{:μ}(), Val{(:out, :τ)}(), (as_vague_msg(NormalMeanVariance), as_vague_msg(Gamma)), nothing, nothing, 1.0, nothing
                )

                io = IOBuffer()
                showerror(io, err)
                output = String(take!(io))

                @test occursin("Possible fix, define:", output)
                @test occursin("@marginalrule", output)
                @test occursin("m_out::NormalMeanVariance", output)
                @test occursin("m_τ::Gamma", output)
                @test occursin("meta::Float64", output)
            end

            let
                err = ReactiveMP.MarginalRuleMethodError(
                    NormalMeanPrecision, Val{:μ}(), nothing, nothing, Val{(:out, :τ)}(), (as_vague_mrg(NormalMeanVariance), as_vague_mrg(Gamma)), 1.0, nothing
                )

                io = IOBuffer()
                showerror(io, err)
                output = String(take!(io))

                @test occursin("Possible fix, define:", output)
                @test occursin("@marginalrule", output)
                @test occursin("q_out::NormalMeanVariance", output)
                @test occursin("q_τ::Gamma", output)
                @test occursin("meta::Float64", output)
            end

            let
                err = ReactiveMP.MarginalRuleMethodError(
                    NormalMeanPrecision, Val{:τ}(), nothing, nothing, Val{(:out_μ,)}(), (Marginal(vague(MvNormalMeanPrecision, 2), false, false, nothing),), 1.0, nothing
                )

                io = IOBuffer()
                showerror(io, err)
                output = String(take!(io))

                @test occursin("Possible fix, define:", output)
                @test occursin("@marginalrule", output)
                @test occursin("q_out_μ::MvNormalMeanPrecision", output)
                @test occursin("meta::Float64", output)
            end

            let
                err = ReactiveMP.MarginalRuleMethodError(
                    NormalMeanPrecision,
                    Val{:τ}(),
                    Val{(:out, :μ)}(),
                    (as_vague_msg(NormalMeanVariance), as_vague_msg(NormalMeanVariance)),
                    Val{(:out_μ,)}(),
                    (Marginal(vague(MvNormalMeanPrecision, 2), false, false, nothing),),
                    1.0,
                    nothing
                )

                io = IOBuffer()
                showerror(io, err)
                output = String(take!(io))

                @test occursin("Possible fix, define", output)
                @test occursin("(m_out::NormalMeanVariance, m_μ::NormalMeanVariance, q_out_μ::MvNormalMeanPrecision, meta::Float64)", output)
            end
        end
    end

    @testset "Check that default meta is `nothing`" begin
        struct DummyNode end
        struct DummyNodeMeta end

        @node DummyNode Stochastic [out, x, y]

        @rule DummyNode(:out, Marginalisation) (m_x::NormalMeanPrecision, m_y::NormalMeanPrecision) = 1
        @rule DummyNode(:out, Marginalisation) (m_x::NormalMeanPrecision, m_y::NormalMeanPrecision, meta::Int) = meta
        @rule DummyNode(:out, Marginalisation) (q_x::NormalMeanPrecision, q_y::NormalMeanPrecision) = 3

        @test (@call_rule DummyNode(:out, Marginalisation) (m_x = vague(NormalMeanPrecision), m_y = vague(NormalMeanPrecision))) === 1

        @test (@call_rule DummyNode(:out, Marginalisation) (m_x = vague(NormalMeanPrecision), m_y = vague(NormalMeanPrecision), meta = nothing)) === 1

        @test (@call_rule DummyNode(:out, Marginalisation) (m_x = vague(NormalMeanPrecision), m_y = vague(NormalMeanPrecision), meta = 2)) === 2

        @test (@call_rule DummyNode(:out, Marginalisation) (m_x = vague(NormalMeanPrecision), m_y = vague(NormalMeanPrecision), meta = 3)) === 3

        @test (@call_rule DummyNode(:out, Marginalisation) (q_x = vague(NormalMeanPrecision), q_y = vague(NormalMeanPrecision))) === 3

        @test (@call_rule DummyNode(:out, Marginalisation) (q_x = vague(NormalMeanPrecision), q_y = vague(NormalMeanPrecision), meta = nothing)) === 3

        @test_throws ReactiveMP.RuleMethodError (@call_rule DummyNode(:out, Marginalisation) (q_x = vague(NormalMeanPrecision), q_y = vague(NormalMeanPrecision), meta = 2))

        @test_throws ReactiveMP.RuleMethodError (@call_rule DummyNode(:out, Marginalisation) (q_x = vague(NormalMeanPrecision), q_y = vague(NormalMeanPrecision), meta = 3))
    end

    @testset "Check the `return_addons` option" begin
        # Enable LogScale addon
        dist_and_addons = @call_rule [return_addons = true] Bernoulli(:out, Marginalisation) (m_p = Beta(1, 2), addons = (AddonLogScale(),))

        @test dist_and_addons isa Tuple
        @test length(dist_and_addons) === 2
        @test dist_and_addons[1] isa Bernoulli
        @test dist_and_addons[2] isa Tuple{AddonLogScale}

        # Without addons but with the option
        dist_and_nothing = @call_rule [return_addons = true] Bernoulli(:out, Marginalisation) (m_p = Beta(1, 2),)

        @test dist_and_nothing isa Tuple
        @test length(dist_and_nothing) === 2
        @test dist_and_nothing[1] isa Bernoulli
        @test dist_and_nothing[2] isa Nothing
    end
end

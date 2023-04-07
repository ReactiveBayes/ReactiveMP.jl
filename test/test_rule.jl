module ReactiveMPRuleTest

using Test
using ReactiveMP
using MacroTools

import MacroTools: inexpr

@testset "rule" begin
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
                @test inexpr(expression, :(ReactiveMP.custom_isapprox))
                @test inexpr(expression, :(ReactiveMP.is_typeof_equal))
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

                    modified_m = :(ReactiveMP.convert_paramfloattype($eltype, $m))
                    modified_v = :(ReactiveMP.convert_paramfloattype($eltype, $v))
                    original_meta = :(meta = Meta(1))
                    modified_meta = :(ReactiveMP.convert_paramfloattype($eltype, Meta(1)))

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
                    test_rules_convert_paramfloattype(:(NormalMeanVariance(1.0, 2.0)), eltype), :(ReactiveMP.convert_paramfloattype($eltype, NormalMeanVariance(1.0, 2.0)))
                )
                @test inexpr(
                    test_rules_convert_paramfloattype(:(m_in = NormalMeanVariance(1.0, 2.0)), eltype),
                    :(m_in = ReactiveMP.convert_paramfloattype($eltype, NormalMeanVariance(1.0, 2.0)))
                )
                @test inexpr(
                    test_rules_convert_paramfloattype(:((m_in = NormalMeanVariance(1.0, 2.0),)), eltype),
                    :((m_in = ReactiveMP.convert_paramfloattype($eltype, NormalMeanVariance(1.0, 2.0)),))
                )
                @test inexpr(
                    test_rules_convert_paramfloattype(:((m_in = ManyOf(NormalMeanVariance(1.0, 2.0)),)), eltype),
                    :((m_in = ManyOf(ReactiveMP.convert_paramfloattype($eltype, NormalMeanVariance(1.0, 2.0))),))
                )
                @test inexpr(
                    test_rules_convert_paramfloattype(:((m_in = NormalMeanVariance(1.0, 2.0), q_out = Gamma(1.0, 2.0))), eltype),
                    :((m_in = ReactiveMP.convert_paramfloattype($eltype, NormalMeanVariance(1.0, 2.0)), q_out = ReactiveMP.convert_paramfloattype($eltype, Gamma(1.0, 2.0))))
                )
            end
        end
    end

    @testset "Macro utilities" begin
        import ReactiveMP: rule_macro_parse_on_tag, rule_macro_parse_fn_args, call_rule_macro_parse_fn_args

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
                    make_node(NormalMeanPrecision)
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
                    make_node(NormalMeanPrecision)
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
                    make_node(NormalMeanPrecision)
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
                    make_node(NormalMeanPrecision)
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
                    make_node(NormalMeanPrecision)
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
                    make_node(NormalMeanPrecision)
                )

                io = IOBuffer()
                showerror(io, err)
                output = String(take!(io))

                @test occursin("[WARN]: Non-standard rule layout found!", output)
                @test occursin("Possible fix, define", output)
            end
        end

        @testset "marginalrule_method_error" begin
            as_vague_msg(::Type{T}) where {T} = Message(vague(T), false, false, nothing)
            as_vague_mrg(::Type{T}) where {T} = Marginal(vague(T), false, false, nothing)

            let
                err = ReactiveMP.MarginalRuleMethodError(
                    NormalMeanPrecision,
                    Val{:μ}(),
                    Val{(:out,)}(),
                    (as_vague_msg(NormalMeanVariance),),
                    Val{(:τ,)}(),
                    (as_vague_mrg(Gamma),),
                    nothing,
                    make_node(NormalMeanPrecision)
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
                    NormalMeanPrecision, Val{:μ}(), Val{(:out,)}(), (as_vague_msg(NormalMeanVariance),), Val{(:τ,)}(), (as_vague_mrg(Gamma),), 1.0, make_node(NormalMeanPrecision)
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
                    NormalMeanPrecision,
                    Val{:μ}(),
                    Val{(:out, :τ)}(),
                    (as_vague_msg(NormalMeanVariance), as_vague_msg(Gamma)),
                    nothing,
                    nothing,
                    1.0,
                    make_node(NormalMeanPrecision)
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
                    NormalMeanPrecision,
                    Val{:μ}(),
                    nothing,
                    nothing,
                    Val{(:out, :τ)}(),
                    (as_vague_mrg(NormalMeanVariance), as_vague_mrg(Gamma)),
                    1.0,
                    make_node(NormalMeanPrecision)
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
                    NormalMeanPrecision,
                    Val{:τ}(),
                    nothing,
                    nothing,
                    Val{(:out_μ,)}(),
                    (Marginal(vague(MvNormalMeanPrecision, 2), false, false, nothing),),
                    1.0,
                    make_node(NormalMeanPrecision)
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
                    make_node(NormalMeanPrecision)
                )

                io = IOBuffer()
                showerror(io, err)
                output = String(take!(io))

                @test occursin("[WARN]: Non-standard rule layout found!", output)
                @test occursin("Possible fix, define", output)
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
end

end

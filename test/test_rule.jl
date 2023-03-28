module ReactiveMPRuleTest

using Test
using ReactiveMP
using MacroTools

import ReactiveMP: rule_macro_parse_on_tag, rule_macro_parse_fn_args, call_rule_macro_parse_fn_args

@testset "rule" begin
    @testset "Testing utilities" begin
        import ReactiveMP: TestRulesConfiguration

        @testset "TestRulesConfiguration" begin
            @test_throws ErrorException TestRulesConfiguration(:([1]))
            @test_throws ErrorException TestRulesConfiguration(:([options = value = broken]))

            # Check that `float_conversions` is set properly
            for conversions in (true, false)
                @test TestRulesConfiguration(:([ with_float_conversions = $conversions ])).with_float_conversions === conversions
            end

            # Check that `tolerance` is set properly
            for atol in (1e-3, 1e-4)
                @test all(value -> value === atol, TestRulesConfiguration(:([atol = $atol])).float_tolerance |> values |> collect)

                @test TestRulesConfiguration(:([float32_atol = $atol])).float_tolerance[Float32] === atol
                @test TestRulesConfiguration(:([float64_atol = $atol])).float_tolerance[Float64] === atol
                @test TestRulesConfiguration(:([bigfloat_atol = $atol])).float_tolerance[BigFloat] === atol
            end
        end
    end

    @testset "Macro utilities" begin
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

module ReactiveMPRuleTest

using Test
using ReactiveMP 
using MacroTools

import ReactiveMP: rule_macro_parse_on_tag, rule_macro_parse_fn_args, call_rule_macro_parse_fn_args

@testset "rule" begin

    @testset "rule_macro_parse_on_tag(expression)" begin
        @test rule_macro_parse_on_tag(:(:out)) == (:(Type{ Val{ :out } }), nothing, nothing)
        @test rule_macro_parse_on_tag(:(:mean)) == (:(Type{ Val{ :mean } }), nothing, nothing)
        @test rule_macro_parse_on_tag(:(:mean, k)) == (:(Tuple{ Val{ :mean }, Int }), :k, :(k = on[2]))
        @test rule_macro_parse_on_tag(:(:precision, r)) == (:(Tuple{ Val{ :precision }, Int }), :r, :(r = on[2]))

        @test_throws ErrorException rule_macro_parse_on_tag(:(out))
        @test_throws ErrorException rule_macro_parse_on_tag(:(123))
        @test_throws ErrorException rule_macro_parse_on_tag(:(:mean, 1))
        @test_throws ErrorException rule_macro_parse_on_tag(:(precision, r))
    end

    @testset "rule_macro_parse_fn_args(inputs; specname, prefix, proxy)" begin
        names, types, init = rule_macro_parse_fn_args([ (:m_out, :PointMass), (:m_mean, :NormalMeanPrecision) ]; specname = :messages, prefix = :m_, proxy = :(ReactiveMP.Message))

        @test names == :(Type{ Val{ (:out, :mean) } })
        @test types == :(Tuple{ ReactiveMP.Message{ <: PointMass }, ReactiveMP.Message{ <: NormalMeanPrecision } })
        @test init == Expr[:(m_out = getdata(messages[1])), :(m_mean = getdata(messages[2]))]

        names, types, init = rule_macro_parse_fn_args([ (:m_out, :PointMass), (:m_mean, :NormalMeanPrecision) ]; specname = :marginals, prefix = :q_, proxy = :(ReactiveMP.Marginal))

        @test names == :Nothing
        @test types == :Nothing
        @test init == Expr[]

        names, types, init = rule_macro_parse_fn_args([ (:m_out, :PointMass), (:q_mean, :NormalMeanPrecision) ]; specname = :marginals, prefix = :q_, proxy = :(ReactiveMP.Marginal))

        @test names == :(Type{ Val{ (:mean, ) } })
        @test types == :(Tuple{ ReactiveMP.Marginal{ <: NormalMeanPrecision }, })
        @test init == Expr[ :(q_mean = getdata(marginals[1])) ]
    end

    @testset "call_rule_macro_parse_fn_args(inputs; specname, prefix, proxy)" begin
        names, values = call_rule_macro_parse_fn_args([ (:m_out, :(PointMass(1.0))), (:m_mean, :(NormalMeanPrecision(0.0, 1.0))) ]; specname = :messages, prefix = :m_, proxy = :(ReactiveMP.Message))

        @test names == :(Val{ (:out, :mean) })
        @test values == :(ReactiveMP.Message(PointMass(1.0), false, false), ReactiveMP.Message(NormalMeanPrecision(0.0, 1.0), false, false))

        names, values = call_rule_macro_parse_fn_args([ (:m_out, :(PointMass(1.0))), (:m_mean, :(NormalMeanPrecision(0.0, 1.0))) ]; specname = :marginals, prefix = :q_, proxy = :(ReactiveMP.Marginal))

        @test names == :nothing
        @test values == :nothing

        names, values = call_rule_macro_parse_fn_args([ (:m_out, :(PointMass(1.0))), (:q_mean, :(NormalMeanPrecision(0.0, 1.0))) ]; specname = :marginals, prefix = :q_, proxy = :(ReactiveMP.Marginal))

        @test names == :(Val{ (:mean, ) })
        @test values == :((ReactiveMP.Marginal(NormalMeanPrecision(0.0, 1.0), false, false),))
    end
    
end

end
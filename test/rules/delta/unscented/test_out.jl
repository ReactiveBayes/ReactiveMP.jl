module RulesDeltaUTOutTest

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
import ReactiveMP: @test_rules

# TODO: check_type_promotion = true breaks

# g: single input, single output
g(x) = x .^ 2 .- 5.0

# g2: same as `g`, but depends on the global variables
t = 2
v = 5.0
g2(x) = x .^ t .- v

# h: multiple input, single output
h(x, y) = x .^ 2 .- y

# g provided in a similar syntax like the N parameter in normal_mixture/test_out.jl
# normal_mixture is the only example with this syntax (that has a test; gamma_mixture is another candidate but ∄ test)
@testset "rules:Delta:unscented:out" begin
    @testset "Single univariate input" begin
        @test_rules [check_type_promotion = false] DeltaFn{g}(:out, Marginalisation) [
            (input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0)), meta = DeltaMeta(; method = Unscented())), output = NormalMeanVariance(2.0000000001164153, 66.00000000093132)),
            (input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0)), meta = DeltaMeta(; method = Unscented(; alpha = 1.0))), output = NormalMeanVariance(2.0, 66.0))
        ]
    end

    @testset "Single multivariate input" begin
        @test_rules [check_type_promotion = false] DeltaFn{g}(:out, Marginalisation) [
            (
                input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0])), meta = DeltaMeta(; method = Unscented())),
                output = MvNormalMeanCovariance([2.0000000001164153], [66.00000000093132])
            ),
            (input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0])), meta = DeltaMeta(; method = Unscented(; alpha = 1.0))), output = MvNormalMeanCovariance([2.0], [66.0]))
        ]
    end

    @testset "Single univariate input" begin
        @test_rules [check_type_promotion = false] DeltaFn{g2}(:out, Marginalisation) [
            (input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0)), meta = DeltaMeta(; method = Unscented())), output = NormalMeanVariance(2.0000000001164153, 66.00000000093132)),
            (input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0)), meta = DeltaMeta(; method = Unscented(; alpha = 1.0))), output = NormalMeanVariance(2.0, 66.0))
        ]
    end

    @testset "Single multivariate input" begin
        @test_rules [check_type_promotion = false] DeltaFn{g2}(:out, Marginalisation) [
            (
                input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0])), meta = DeltaMeta(; method = Unscented())),
                output = MvNormalMeanCovariance([2.0000000001164153], [66.00000000093132])
            ),
            (input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0])), meta = DeltaMeta(; method = Unscented(; alpha = 1.0))), output = MvNormalMeanCovariance([2.0], [66.0]))
        ]
    end

    @testset "Multiple univariate input" begin
        @test_rules [check_type_promotion = false] DeltaFn{h}(:out, Marginalisation) [(
            input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0), NormalMeanVariance(5.0, 1.0)), meta = DeltaMeta(; method = Unscented())),
            output = NormalMeanVariance(1.9999999997671694, 67.00000899657607)
        )]
    end

    @testset "Multiple multivariate input" begin
        @test_rules [check_type_promotion = false] DeltaFn{h}(:out, Marginalisation) [(
            input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0]), MvNormalMeanCovariance([5.0], [1.0])), meta = DeltaMeta(; method = Unscented())),
            output = MvNormalMeanCovariance([1.9999999997671694], [67.00000899657607])
        )]
    end
end
end

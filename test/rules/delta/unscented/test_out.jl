module RulesDeltaUTOutTest

using Test
using ReactiveMP
import ReactiveMP: @test_rules

# TODO: with_float_conversions = true breaks

# g: single input, single output
g(x) = x .^ 2 .- 5.0

# h: multiple input, single output
h(x, y) = x .^ 2 .- y

# g provided in a similar syntax like the N parameter in normal_mixture/test_out.jl
# normal_mixture is the only example with this syntax (that has a test; gamma_mixture is another candidate but ∄ test)
@testset "rules:Delta:unscented:out" begin
    @testset "Single univariate input" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}(:out, Marginalisation) [
            (
                input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0)), meta = DeltaUnscented()),
                output = NormalMeanVariance(2.0000000001164153, 66.00000000093132)
            ),
            (
                input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0)), meta = DeltaUnscented(alpha = 1.0)),
                output = NormalMeanVariance(2.0, 66.0)
            )
        ]
    end

    @testset "Single multivariate input" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}(:out, Marginalisation) [
            (
                input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0])), meta = DeltaUnscented()),
                output = MvNormalMeanCovariance([2.0000000001164153], [66.00000000093132])
            ),
            (
                input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0])), meta = DeltaUnscented(alpha = 1.0)),
                output = MvNormalMeanCovariance([2.0], [66.0])
            )
        ]
    end

    @testset "Multiple univariate input" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}(:out, Marginalisation) [
            (
            input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0), NormalMeanVariance(5.0, 1.0)), meta = DeltaUnscented()),
            output = NormalMeanVariance(1.9999999997671694, 67.00000899657607)
        )
        ]
    end

    @testset "Multiple multivariate input" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}(:out, Marginalisation) [
            (
            input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0]), MvNormalMeanCovariance([5.0], [1.0])), meta = DeltaUnscented()),
            output = MvNormalMeanCovariance([1.9999999997671694], [67.00000899657607])
        )
        ]
    end
end
end

module RulesDeltaETMarginalsTest

using Test
using ReactiveMP
import ReactiveMP: @test_marginalrules

# TODO: with_float_conversions = true breaks
# TODO: only tests for multiple input, single output with unknown inverse exist

# g: single input, single output
#g(x::Number) = x^2 - 5.0
#g(x::Vector) = x .^ 2 .- 5.0

# h: multiple input, single output
h(x::Number, y::Number) = x^2 - y
h(x::Vector, y::Vector) = x .^ 2 .- y

@testset "rules:Delta:extended:marginals" begin
    @testset "Marginal: f(x) (m_ins::NormalMeanVariance, meta.inverse::Nothing)" begin
        # ForneyLab:test_delta_extended:MDeltaEInGX 1
        @test_marginalrules [with_float_conversions = false] DeltaFn{h}(:ins) [
            (
            input = (
                m_out = NormalMeanVariance(2.0, 3.0),
                m_ins = ManyOf(NormalMeanVariance(2.0, 1.0), NormalMeanVariance(5.0, 1.0)),
                meta  = DeltaExtended(inverse = nothing)
            ),
            output = DeltaMarginal(
                MvNormalMeanCovariance(
                    [2.6, 4.85],
                    [0.20000000000000007 0.19999999999999998; 0.19999999999999998 0.95]
                ),
                Any[(), ()] # [1, 1] # TODO: dimensions "ds" are not correct for the left marginal
            )
        )
        ]
    end

    @testset "Marginal: f(x) (m_ins::MvNormalMeanCovariance, meta.inverse::Nothing)" begin
        # ForneyLab:test_delta_extended:MDeltaEInGX 2
        @test_marginalrules [with_float_conversions = false] DeltaFn{h}(:ins) [
            (
            input = (
                m_out = MvNormalMeanCovariance([2.0], [3.0]),
                m_ins = ManyOf(MvNormalMeanCovariance([2.0], [1.0]), MvNormalMeanCovariance([5.0], [1.0])),
                meta  = DeltaExtended(inverse = nothing)
            ),
            output = DeltaMarginal(
                MvNormalMeanCovariance(
                    [2.6, 4.85],
                    [0.20000000000000007 0.19999999999999998; 0.19999999999999998 0.95]
                ),
                Any[(1,), (1,)] # [1, 1] # TODO: dimensions "ds" are not correct for the left marginal
            )
        )
        ]
    end
end # testset
end # module

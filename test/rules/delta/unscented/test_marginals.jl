module RulesDeltaUTMarginalsTest

using Test
using ReactiveMP
import ReactiveMP: @test_marginalrules

# TODO: with_float_conversions = true breaks
# TODO: only tests for multiple input, single output with unknown inverse exist

# g: single input, single output
#g(x::Float64) = x^2 - 5.0
#g(x::Vector{Float64}) = x .^ 2 .- 5.0

# h: multiple input, single output
h(x::Float64, y::Float64) = x^2 - y
h(x::Vector{Float64}, y::Vector{Float64}) = x .^ 2 .- y
h(x::Float64, y::Vector{Float64}) = x^2 .- y

@testset "rules:Delta:unscented:marginals" begin
    @testset "Marginal: f(x) (m_ins::NormalMeanVariance, meta.inverse::Nothing)" begin
        # ForneyLab:test_delta_unscented:MDeltaUTInGX 1
        @test_marginalrules [with_float_conversions = false] DeltaFn{h}(:ins) [
            (
            input = (
                m_out = NormalMeanVariance(2.0, 3.0),
                m_ins = ManyOf(NormalMeanVariance(2.0, 1.0), NormalMeanVariance(5.0, 1.0)),
                meta  = DeltaUnscented(inverse = nothing)
            ),
            output = DeltaMarginal( # TODO: Seems like the test result in ForneyLab is wrong? Should be the same as the test result below
                MvNormalMeanCovariance(
                    # vs left: [2.3636363470609245, 4.909090913233555]
                    [2.3636363470614055, 4.9090909132334355],
                    [0.2727273058237252 0.1818181735464949; 0.18181817354649488 0.9545454566127697]
                    # vs left: [0.2727273058246874 0.18181817354625435; 0.18181817354625435 0.9545454566128299]
                ),
                Any[(), ()] # [2] # TODO: dimensions "ds" are not correct for the left marginal
            )
        )
        ]
    end

    @testset "Marginal: f(x) (m_ins::MvNormalMeanCovariance, meta.inverse::Nothing)" begin
        # ForneyLab:test_delta_unscented:MDeltaUTInGX 2
        @test_marginalrules [with_float_conversions = false] DeltaFn{h}(:ins) [
            (
            input = (
                m_out = MvNormalMeanCovariance([2.0], [3.0]),
                m_ins = ManyOf(MvNormalMeanCovariance([2.0], [1.0]), MvNormalMeanCovariance([5.0], [1.0])),
                meta  = DeltaUnscented(inverse = nothing)
            ),
            output = DeltaMarginal(
                MvNormalMeanCovariance(
                    [2.3636363470609245, 4.909090913233555],
                    [0.2727273058246874 0.18181817354625435; 0.18181817354625435 0.9545454566128299]
                ),
                Any[(1,), (1,)] # [1, 1] # TODO: dimensions "ds" are not correct for the left marginal:
            )
        )
        ]
    end
end # testset
end # module

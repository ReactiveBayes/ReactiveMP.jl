module RulesDeltaUTMarginalsTest

using Test
using ReactiveMP
import ReactiveMP: @test_marginalrules

# g: single input, single output
g(x) = x .^ 2 .- 5.0

# h: multiple input, single output
h(x, y) = x .^ 2 .- y

@testset "rules:Delta:unscented:marginals" begin
    @testset "Single univariate input" begin
        @test_marginalrules [with_float_conversions = false, atol = 1e-10] DeltaFn{g}(:ins) [
            (
            input = (
                m_out = NormalMeanVariance(2.0, 3.0),
                m_ins = ManyOf(NormalMeanVariance(2.0, 1.0)),
                meta  = DeltaUnscented(inverse = nothing)
            ),
            output = DeltaMarginal(
                NormalMeanVariance(
                    2.3809523807887425,
                    0.23809523822182999
                ),
                [(1,)]
            )
        )
        ]
    end

    @testset "Single multivariate input" begin
        @test_marginalrules [with_float_conversions = false] DeltaFn{g}(:ins) [
            (
            input = (
                m_out = MvNormalMeanCovariance([2.0], [3.0]),
                m_ins = ManyOf(MvNormalMeanCovariance([2.0], [1.0;;])),
                meta  = DeltaUnscented(inverse = nothing)
            ),
            output = DeltaMarginal(
                MvNormalMeanCovariance(
                    [2.3809523807887425],
                    [0.23809523822182999;;]
                ),
                [(1,)]
            )
        )
        ]
    end

    @testset "Multiple univairate input" begin
        # ForneyLab:test_delta_unscented:MDeltaUTInGX 1
        @test_marginalrules [with_float_conversions = false, atol = 1e-4] DeltaFn{h}(:ins) [
            (
            input = (
                m_out = NormalMeanVariance(2.0, 3.0),
                m_ins = ManyOf(NormalMeanVariance(2.0, 1.0), NormalMeanVariance(5.0, 1.0)),
                meta  = DeltaUnscented(inverse = nothing)
            ),
            output = DeltaMarginal(
                MvNormalMeanCovariance(
                    # vs left: [2.3636363470609245, 4.909090913233555]
                    [2.3636363470614055, 4.9090909132334355],
                    [0.2727273058237252 0.1818181735464949; 0.18181817354649488 0.9545454566127697]
                    # vs left: [0.2727273058246874 0.18181817354625435; 0.18181817354625435 0.9545454566128299]
                ),
                Any[(), ()]
            )
        )
        ]
    end

    @testset "Multiple multivariate input" begin
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
                Any[(1,), (1,)] # [1, 1]
            )
        )
        ]
    end
end # testset
end # module

module RulesDeltaETInTest

using Test
using Random
using ReactiveMP
import ReactiveMP: @test_rules

Random.seed!(11)

# TODO: with_float_conversions = true breaks

# g: single input, single output
g(x) = x .^ 2 .- 5.0
g_inv(y) = sqrt.(y .+ 5.0)

# h: multiple input, single output
h(x, y) = x .^ 2 .- y
h_inv_x(z, y) = sqrt.(z .+ y)
h_inv_z(x, y) = x .^ 2 .- y

@testset "rules:Delta:extended:in" begin
    @testset "Single input with known inverse" begin
        @test_rules [with_float_conversions = false, atol = 1e-5] DeltaFn{g}((:in, k = 1), Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(2.0, 3.0), m_ins = nothing, meta = DeltaLinearization(inverse = g_inv)),
                output = NormalMeanVariance(2.6457513110645907, 0.10714285714285711)
            ),
            (
                input = (m_out = MvNormalMeanCovariance([2.0], [3.0;;]), m_ins = nothing, meta = DeltaLinearization(inverse = g_inv)),
                output = MvNormalMeanCovariance([2.6457513110645907], [0.10714285714285711;;])
            )
        ]
    end
    @testset "Multiple input with known inverse" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}((:in, k = 1), Marginalisation) [
            (
                input = (
                    m_out = NormalMeanVariance(2.0, 3.0),
                    m_ins = ManyOf(NormalMeanVariance(5.0, 1.0)),
                    meta = DeltaLinearization(inverse = (h_inv_x, h_inv_z))
                ),
                output = NormalMeanVariance(2.6457513110645907, 0.14285714285714282)
            ),
            (
                input = (m_out = MvNormalMeanCovariance([2.0], [3.0;;]),
                    m_ins = ManyOf(MvNormalMeanCovariance([5.0], [1.0;;])), meta = DeltaLinearization(inverse = (h_inv_x, h_inv_z))),
                output = MvNormalMeanCovariance([2.6457513110645907], [0.14285714285714282;;])
            )
        ]
        @test_rules [with_float_conversions = false, atol = 1e-5] DeltaFn{h}((:in, k = 2), Marginalisation) [
            (
                input = (
                    m_out = NormalMeanVariance(2.0, 1.0),
                    m_ins = ManyOf(NormalMeanVariance(5.0, 1.0)),
                    meta = DeltaLinearization(inverse = (h_inv_x, h_inv_z))
                ),
                output = NormalMeanVariance(-1.0, 17)
            ),
            (
                input = (m_out = MvNormalMeanCovariance([2.0], [1.0]),
                    m_ins = ManyOf(MvNormalMeanCovariance([5.0], [1.0])), meta = DeltaLinearization(inverse = (h_inv_x, h_inv_z))),
                output = MvNormalMeanCovariance([-1.0], [17.0;;])
            )
        ]
    end

    @testset "Single input with unknown inverse" begin
        @test_rules [with_float_conversions = false, atol = 1e-3] DeltaFn{h}((:in, k = 1), Marginalisation) [
            (
                input = (
                    q_ins = DeltaMarginal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), [(), ()]),
                    m_in = NormalMeanVariance(5.0, 10.0),
                    meta = DeltaLinearization()
                ),
                output = NormalWeightedMeanPrecision(0.5, 0.9)
            ),
            (
                input = (q_ins = DeltaMarginal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), [(1,), (1,)]),
                    m_in = MvNormalMeanCovariance([5.0], [10.0;;]), meta = DeltaLinearization()),
                output = MvNormalWeightedMeanPrecision([0.5], [0.9;;])
            )
        ]
    end
    @testset "Multiple input with unknown inverse" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}((:in, k = 2), Marginalisation) [
            (
                input = (
                    q_ins = DeltaMarginal(MvNormalMeanCovariance(ones(3), diageye(3)), [(), (), ()]),
                    m_in = NormalMeanVariance(0.0, 10.0),
                    meta = DeltaLinearization()
                ),
                output = NormalWeightedMeanPrecision(1.0, 0.9)
            ),
            (
                input = (
                    q_ins = DeltaMarginal(MvNormalMeanCovariance(ones(3), diageye(3)), [(1,), (2,), ()]),
                    m_in = MvNormalMeanCovariance(zeros(2), 10 * diageye(2)),
                    meta = DeltaLinearization()
                ),
                output = MvNormalWeightedMeanPrecision(ones(2), 0.9 * diageye(2))
            )
        ]
    end
end # testset
end # module

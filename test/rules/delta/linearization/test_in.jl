module RulesDeltaETInTest

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
import ReactiveMP: @test_rules

# g: single input, single output
g(x) = x .^ 2 .- 5
g_inv(y) = sqrt.(y .+ 5)

# h: multiple input, single output
h(x, y) = x .^ 2 .- y
h_inv_x(z, y) = sqrt.(z .+ y)
h_inv_z(x, y) = x .^ 2 .- y

@testset "rules:Delta:linearization:in" begin
    @testset "Single input with known inverse" begin
        @test_rules [check_type_promotion = true, atol = 1e-5] DeltaFn{g}((:in, k = 1), Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(2.0, 3.0), m_ins = nothing, meta = DeltaMeta(; method = Linearization(), inverse = g_inv)),
                output = NormalMeanVariance(2.6457513110645907, 0.10714285714285711)
            ),
            (
                input = (m_out = MvNormalMeanCovariance([2.0], [3.0;;]), m_ins = nothing, meta = DeltaMeta(; method = Linearization(), inverse = g_inv)),
                output = MvNormalMeanCovariance([2.6457513110645907], [0.10714285714285711;;])
            )
        ]
    end

    @testset "Multiple input with known inverse" begin
        @test_rules [check_type_promotion = true] DeltaFn{h}((:in, k = 1), Marginalisation) [
            (
                input = (
                    m_out = NormalMeanVariance(2.0, 3.0), m_ins = ManyOf(NormalMeanVariance(5.0, 1.0)), meta = DeltaMeta(; method = Linearization(), inverse = (h_inv_x, h_inv_z))
                ),
                output = NormalMeanVariance(2.6457513110645907, 0.14285714285714282)
            ),
            (
                input = (
                    m_out = MvNormalMeanCovariance([2.0], [3.0;;]),
                    m_ins = ManyOf(MvNormalMeanCovariance([5.0], [1.0;;])),
                    meta  = DeltaMeta(; method = Linearization(), inverse = (h_inv_x, h_inv_z))
                ),
                output = MvNormalMeanCovariance([2.6457513110645907], [0.14285714285714282;;])
            )
        ]
        @test_rules [check_type_promotion = true, atol = 1e-5] DeltaFn{h}((:in, k = 2), Marginalisation) [
            (
                input = (
                    m_out = NormalMeanVariance(2.0, 1.0), m_ins = ManyOf(NormalMeanVariance(5.0, 1.0)), meta = DeltaMeta(; method = Linearization(), inverse = (h_inv_x, h_inv_z))
                ),
                output = NormalMeanVariance(-1.0, 17)
            ),
            (
                input = (
                    m_out = MvNormalMeanCovariance([2.0], [1.0]),
                    m_ins = ManyOf(MvNormalMeanCovariance([5.0], [1.0])),
                    meta = DeltaMeta(; method = Linearization(), inverse = (h_inv_x, h_inv_z))
                ),
                output = MvNormalMeanCovariance([-1.0], [17.0;;])
            )
        ]
    end

    @testset "Single input with unknown inverse" begin
        @test_rules [check_type_promotion = true, atol = 1e-3] DeltaFn{h}((:in, k = 1), Marginalisation) [
            (
                input = (
                    q_ins = JointNormal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), ((), ())),
                    m_in = NormalMeanVariance(5.0, 10.0),
                    meta = DeltaMeta(; method = Linearization())
                ),
                output = NormalWeightedMeanPrecision(0.5, 0.9)
            ),
            (
                input = (
                    q_ins = JointNormal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), ((1,), (1,))),
                    m_in = MvNormalMeanCovariance([5.0], [10.0;;]),
                    meta = DeltaMeta(; method = Linearization())
                ),
                output = MvNormalWeightedMeanPrecision([0.5], [0.9;;])
            )
        ]
    end

    @testset "Multiple input with unknown inverse" begin
        @test_rules [check_type_promotion = true] DeltaFn{h}((:in, k = 2), Marginalisation) [
            (
                input = (
                    q_ins = JointNormal(MvNormalMeanCovariance(ones(3), diageye(3)), ((), (), ())),
                    m_in = NormalMeanVariance(0.0, 10.0),
                    meta = DeltaMeta(; method = Linearization())
                ),
                output = NormalWeightedMeanPrecision(1.0, 0.9)
            ),
            (
                input = (
                    q_ins = JointNormal(MvNormalMeanCovariance(ones(3), diageye(3)), ((1,), (2,), ())),
                    m_in = MvNormalMeanCovariance(zeros(2), 10 * diageye(2)),
                    meta = DeltaMeta(; method = Linearization())
                ),
                output = MvNormalWeightedMeanPrecision(ones(2), 0.9 * diageye(2))
            )
        ]
    end
end # testset
end # module

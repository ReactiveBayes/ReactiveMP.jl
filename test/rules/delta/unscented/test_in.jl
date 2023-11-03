module RulesDeltaUTInTest

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
import ReactiveMP: @test_rules

# TODO: check_type_promotion = true breaks

# g: single input, single output
g(x) = x .^ 2 .- 5.0
g_inv(y) = sqrt.(y .+ 5.0)

# h: multiple input, single output
h(x, y) = x .^ 2 .- y
h_inv_x(z, y) = sqrt.(z .+ y)
h_inv_z(x, y) = x .^ 2 .- y

@testset "rules:Delta:unscented:in" begin
    @testset "Single input with known inverse" begin
        @test_rules [check_type_promotion = false] DeltaFn{g}((:in, k = 1), Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(2.0, 3.0), m_ins = nothing, meta = DeltaMeta(; method = Unscented(), inverse = g_inv)),
                output = NormalMeanVariance(2.6255032138433307, 0.10796282966583703)
            ),
            (
                input = (m_out = MvNormalMeanCovariance([2.0], [3.0;;]), m_ins = nothing, meta = DeltaMeta(; method = Unscented(), inverse = g_inv)),
                output = MvNormalMeanCovariance([2.6255032138433307], [0.10796282966583703;;])
            )
        ]
    end

    @testset "Multiple input with known inverse" begin
        @test_rules [check_type_promotion = false] DeltaFn{h}((:in, k = 1), Marginalisation) [
            (
                input = (
                    m_out = NormalMeanVariance(2.0, 3.0), m_ins = ManyOf(NormalMeanVariance(5.0, 1.0)), meta = DeltaMeta(; method = Unscented(), inverse = (h_inv_x, h_inv_z))
                ),
                output = NormalMeanVariance(2.6187538476660848, 0.14431487274498522)
            ),
            (
                input = (
                    m_out = MvNormalMeanCovariance([2.0], [3.0;;]),
                    m_ins = ManyOf(MvNormalMeanCovariance([5.0], [1.0;;])),
                    meta = DeltaMeta(; method = Unscented(), inverse = (h_inv_x, h_inv_z))
                ),
                output = MvNormalMeanCovariance([2.6187538476660848], [0.14431487274498522;;])
            )
        ]

        @test_rules [check_type_promotion = false] DeltaFn{h}((:in, k = 2), Marginalisation) [
            (
                input = (
                    m_out = NormalMeanVariance(2.0, 1.0), m_ins = ManyOf(NormalMeanVariance(3.0, 1.0)), meta = DeltaMeta(; method = Unscented(), inverse = (h_inv_x, h_inv_z))
                ),
                output = NormalMeanVariance(2.0000000002328306, 19.00000100088073)
            ),
            (
                input = (
                    m_out = MvNormalMeanCovariance([2.0], [1.0]),
                    m_ins = ManyOf(MvNormalMeanCovariance([3.0], [1.0])),
                    meta = DeltaMeta(; method = Unscented(), inverse = (h_inv_x, h_inv_z))
                ),
                output = MvNormalMeanCovariance([2.0000000002328306], [19.00000100088073;;])
            )
        ]
    end

    @testset "Single input with unknown inverse" begin
        @test_rules [check_type_promotion = false, atol = 1e-3] DeltaFn{h}((:in, k = 1), Marginalisation) [
            (
                input = (
                    q_ins = JointNormal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), ((), ())),
                    m_in = NormalMeanVariance(5.0, 10.0),
                    meta = DeltaMeta(; method = Unscented())
                ),
                output = NormalWeightedMeanPrecision(0.5, 0.9)
            ),
            (
                input = (
                    q_ins = JointNormal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), ((1,), (1,))),
                    m_in = MvNormalMeanCovariance([5.0], [10.0;;]),
                    meta = DeltaMeta(; method = Unscented())
                ),
                output = MvNormalWeightedMeanPrecision([0.5], [0.9;;])
            )
        ]
    end

    @testset "Multiple input with unknown inverse" begin
        @test_rules [check_type_promotion = false] DeltaFn{h}((:in, k = 2), Marginalisation) [
            (
                input = (
                    q_ins = JointNormal(MvNormalMeanCovariance(ones(3), diageye(3)), ((), (), ())), m_in = NormalMeanVariance(0.0, 10.0), meta = DeltaMeta(; method = Unscented())
                ),
                output = NormalWeightedMeanPrecision(1.0, 0.9)
            ),
            (
                input = (
                    q_ins = JointNormal(MvNormalMeanCovariance(ones(3), diageye(3)), ((1,), (2,), ())),
                    m_in = MvNormalMeanCovariance(zeros(2), 10 * diageye(2)),
                    meta = DeltaMeta(; method = Unscented())
                ),
                output = MvNormalWeightedMeanPrecision(ones(2), 0.9 * diageye(2))
            )
        ]
    end
end # testset
end # module

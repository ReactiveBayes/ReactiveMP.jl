module RulesDeltaETOutTest

using Test
using ReactiveMP
import ReactiveMP: @test_rules

# TODO: with_float_conversions = true breaks

# g: single input, single output
g(x) = x .^ 2 .- 5.0

# g2: same as `g`, but depends on the global variables
t = 2
v = 5.0
g2(x) = x .^ t .- v

# h: multiple input, single output
h(x, y) = x .^ 2 .- y

@testset "rules:Delta:extended:out" begin
    # ForneyLab:test_delta_extended:SPDeltaEOutNG 1
    @testset "Belief Propagation: f(x) (m_ins::NormalMeanVariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}(:out, Marginalisation) [(
            input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0)), meta = DeltaMeta(; method = Linearization(), inverse = nothing)), output = NormalMeanVariance(-1.0, 48.0)
        )]
    end

    # ForneyLab:test_delta_extended:SPDeltaEOutNG 2
    @testset "Belief Propagation: f(x): (m_ins::MvNormalMeanCovariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}(:out, Marginalisation) [(
            input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0])), meta = DeltaMeta(; method = Linearization())), output = MvNormalMeanCovariance([-1.0], [48.0])
        )]
    end

    @testset "Belief Propagation: f(x) (m_ins::NormalMeanVariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g2}(:out, Marginalisation) [(
            input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0)), meta = DeltaMeta(; method = Linearization(), inverse = nothing)), output = NormalMeanVariance(-1.0, 48.0)
        )]
    end

    # ForneyLab:test_delta_extended:SPDeltaEOutNG 2
    @testset "Belief Propagation: f(x): (m_ins::MvNormalMeanCovariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g2}(:out, Marginalisation) [(
            input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0])), meta = DeltaMeta(; method = Linearization())), output = MvNormalMeanCovariance([-1.0], [48.0])
        )]
    end

    # ForneyLab:test_delta_extended:SPDeltaEOutNGX 1
    @testset "Belief Propagation: f(x,y) (m_ins::NormalMeanVariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}(:out, Marginalisation) [(
            input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0), NormalMeanVariance(5.0, 1.0)), meta = DeltaMeta(; method = Linearization())),
            output = NormalMeanVariance(-1.0, 49.0)
        )]
    end

    # ForneyLab:test_delta_extended:SPDeltaEOutNGX 2
    @testset "Belief Propagation: f(x,y) (m_ins::MvNormalMeanCovariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}(:out, Marginalisation) [(
            input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0]), MvNormalMeanCovariance([5.0], [1.0])), meta = DeltaMeta(; method = Linearization())),
            output = MvNormalMeanCovariance([-1.0], [49.0])
        )]
    end
end
end

module RulesMultiplicationOutTest

using Test
using ReactiveMP
using Random, Distributions, StableRNGs
import ReactiveMP: @test_rules, besselmod, make_productdist_message

@testset "rule:typeof(*):out" begin
    @testset "Univariate Gaussian messages" begin
        rng = StableRNG(42)
        d1 = NormalMeanVariance(0.0, 1.0)
        d2 = NormalMeanVariance(0.5, 1.5)
        d3 = NormalMeanVariance(2.0, 0.5)
        OutMessage_1 = @call_rule typeof(*)(:out, Marginalisation) (m_A = d1, m_in = d2, meta = TinyCorrection())
        OutMessage_2 = @call_rule typeof(*)(:out, Marginalisation) (m_A = d1, m_in = d3, meta = TinyCorrection())
        OutMessage_3 = @call_rule typeof(*)(:out, Marginalisation) (m_A = d2, m_in = d3, meta = TinyCorrection())
        groundtruthOutMessage_1 = besselmod(mean(d1), var(d1), mean(d2), var(d2), 0.0)
        groundtruthOutMessage_2 = besselmod(mean(d1), var(d1), mean(d3), var(d3), 0.0)
        groundtruthOutMessage_3 = besselmod(mean(d2), var(d2), mean(d3), var(d3), 0.0)

        @test typeof(OutMessage_1) <: ContinuousUnivariateLogPdf
        @test typeof(OutMessage_2) <: ContinuousUnivariateLogPdf
        @test typeof(OutMessage_3) <: ContinuousUnivariateLogPdf

        samples = rand(rng, Uniform(0.5, 4), 10)
        for i in samples
            @test OutMessage_1(i) ≈ groundtruthOutMessage_1(i)
            @test OutMessage_2(i) ≈ groundtruthOutMessage_2(i)
            @test OutMessage_3(i) ≈ groundtruthOutMessage_3(i)
        end
    end
    @testset "messages of type Any" begin
        rng = StableRNG(42)
        d1          = NormalMeanVariance(0.0, 1.0)
        d2          = LogNormal(0.0, 1.0)
        num_samples = 3000
        samples_d1  = rand(rng, d1, num_samples)

        OutMessage = @call_rule typeof(*)(:out, Marginalisation) (m_A = d1, m_in = d2, meta = TinyCorrection())

        @test typeof(OutMessage) <: ContinuousUnivariateLogPdf

        groundtruthOutMessage = make_productdist_message(samples_d1, d2)
        @test isapprox(OutMessage(1.0), groundtruthOutMessage(1.0); atol = 0.1)
        @test isapprox(OutMessage(2.0), groundtruthOutMessage(2.0); atol = 0.1)
        @test isapprox(OutMessage(3.1), groundtruthOutMessage(3.1); atol = 0.1)
    end
end

end

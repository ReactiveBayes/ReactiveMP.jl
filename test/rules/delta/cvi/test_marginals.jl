module RulesCVIMarginalsTest

using Test
using ReactiveMP
using Random
using Distributions
using Flux

id(x) = x

@testset "marginalrules:CVI" begin
    @testset "[id(x) - y] x~Normal, y~Normal" begin
        seed = 123
        rng = MersenneTwister(seed)
        optimizer = Descent(0.01)
        test_meta = CVIApproximation(rng, 1, 500, optimizer)
        
        m_in = NormalMeanVariance()
        marginals = @call_marginalrule DeltaFn{id}(:ins) (m_out = NormalMeanVariance(1, 1), m_ins = ManyOf(m_in,), meta = test_meta)
        result_marginal = convert(Distribution, naturalparams(marginals[1]) - naturalparams(m_in))
        @test isapprox(NormalMeanVariance(mean(result_marginal), var(result_marginal)), NormalMeanVariance(1, 1), atol = 0.1)
    end
end
end
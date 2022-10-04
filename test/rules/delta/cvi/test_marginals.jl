module RulesCVIMarginalsTest

using Test
using ReactiveMP
using Random
using Distributions
using Flux

import ReactiveMP: @test_marginalrules

id(x) = x

@testset "marginalrules:CVI" begin
    @testset "[id(x) - y] x~Normal, y~Normal" begin
        seed = 123
        rng = MersenneTwister(seed)
        optimizer = Descent(0.01)
        test_meta = CVIApproximation(rng, 1, 500, optimizer)

        m_in = NormalMeanVariance()
        m_out = NormalMeanVariance(1, 1)
        marginals = @call_marginalrule DeltaFn{id}(:ins) (m_out = NormalMeanVariance(1, 1), m_ins = ManyOf(m_in), meta = test_meta)

        @show marginals
        @show naturalparams(marginals[1])
        @show naturalparams(m_in) + naturalparams(m_out)

        result_marginal = convert(Distribution, naturalparams(marginals[1]) - naturalparams(m_in))

        @test_marginalrules DeltaFn{id}(:ins) [
            (
                input = (m_out = NormalMeanVariance(1, 1), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta),
                output = (out = convert(Distribution, naturalparams(m_in) + naturalparams(m_out)),)
            )
        ]
    end
end
end

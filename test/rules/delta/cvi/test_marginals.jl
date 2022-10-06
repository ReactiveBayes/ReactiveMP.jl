module RulesCVIMarginalsTest

using Test
using ReactiveMP
using Random
using Distributions
using Flux
using StableRNGs

import ReactiveMP: @test_marginalrules

id(x) = x

@testset "marginalrules:CVI" begin
    @testset "[id(x) - y] x~Normal, y~Normal" begin
        seed = 123
        rng = StableRNG(seed)
        optimizer = Descent(0.01)
        test_meta = CVIApproximation(rng, 1, 500, optimizer)
        @test_marginalrules [with_float_conversions = false, atol = 0.1] DeltaFn{id}(:ins) [
            (
                input = (m_out = NormalMeanVariance(1, 1), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta),
                output = FactorProduct((NormalWeightedMeanPrecision(1.0, 2.0),))
            ),
            (
                input = (m_out = NormalMeanVariance(2, 1), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta),
                output = FactorProduct((NormalWeightedMeanPrecision(2.0, 2.0),))
            ),
            (
                input = (m_out = NormalMeanVariance(10, 1), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta),
                output = FactorProduct((NormalWeightedMeanPrecision(10.0, 2.0),))
            )
        ]
    end
end
end

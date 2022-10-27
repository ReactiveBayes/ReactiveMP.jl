module RulesDeltaCVIInTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules
import ReactiveMP: FactorizedJoint

g(x) = x

struct EmptyOptimizer end

@testset "rules:Delta:cvi:in" begin
    @test_rules [with_float_conversions = false] DeltaFn{g}((:in, k = 1), Marginalisation) [
        (
        input = (
            q_ins = FactorizedJoint((NormalMeanVariance(),)),
            m_in = NormalMeanVariance(1, 2),
            meta = CVIApproximation(1, 1, EmptyOptimizer())
        ),
        output = NormalWeightedMeanPrecision(-0.5, 0.5)
    )
    ]
end # testset
end # module

module RulesDeltaCVIMarginalsTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules

g(x) = x

struct EmptyOptimizer end

# test this set with $ make test testset='rules:gamma_inverse:out'
@testset "rules:Delta:cvi:marginals" begin
    @test_rules [with_float_conversions = false] DeltaFn{g}((:in, k = 1), Marginalisation) [
        (
        input = (
            q_ins = FactorProduct((NormalMeanVariance(),)),
            m_in = NormalMeanVariance(1, 2),
            meta = CVIApproximation(1, 1, EmptyOptimizer)
        ),
        output = NormalWeightedMeanPrecision(-0.5, 0.5)
    )
    ]
end # testset
end # module

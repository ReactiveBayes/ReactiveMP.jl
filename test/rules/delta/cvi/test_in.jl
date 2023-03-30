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
    @test_rules [check_type_promotion = false] DeltaFn{g}((:in, k = 1), Marginalisation) [
        (
            input = (q_ins = FactorizedJoint((NormalMeanVariance(),)), m_in = NormalMeanVariance(1, 2), meta = DeltaMeta(method = CVI(1, 1, EmptyOptimizer()))),
            output = NormalWeightedMeanPrecision(-0.5, 0.5)
        ),
        (
            input = (q_ins = FactorizedJoint((GammaShapeRate(2, 2),)), m_in = GammaShapeRate(1, 1), meta = DeltaMeta(method = CVI(1, 1, EmptyOptimizer()))),
            output = GammaShapeRate(2.0, 1.0)
        )
    ]

    @test_rules [check_type_promotion = false] DeltaFn{g}((:in, k = 2), Marginalisation) [
        (
            input = (
                q_ins = FactorizedJoint((GammaShapeRate(2, 2), NormalMeanVariance())), m_in = NormalMeanVariance(1, 2), meta = DeltaMeta(method = CVI(1, 1, EmptyOptimizer()))
            ),
            output = NormalWeightedMeanPrecision(-0.5, 0.5)
        ),
        (
            input = (q_ins = FactorizedJoint((NormalMeanVariance(), GammaShapeRate(2, 2))), m_in = GammaShapeRate(1, 1), meta = DeltaMeta(method = CVI(1, 1, EmptyOptimizer()))),
            output = GammaShapeRate(2.0, 1.0)
        )
    ]
end # testset
end # module

module RulesDeltaCVIInTest

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
import ReactiveMP: @test_rules

g(x) = x

struct EmptyOptimizer end

@testset "rules:Delta:cvi:in" begin
    @test_rules [check_type_promotion = true] DeltaFn{g}((:in, k = 1), Marginalisation) [
        (
            input = (q_ins = FactorizedJoint((NormalMeanVariance(),)), m_in = NormalMeanVariance(1, 2), meta = DeltaMeta(method = CVI(1, 1, EmptyOptimizer()))),
            output = NormalMeanVariance(-1.0, 2.0)
        ),
        (
            input = (q_ins = FactorizedJoint((GammaShapeRate(2, 2),)), m_in = GammaShapeRate(1, 1), meta = DeltaMeta(method = CVI(1, 1, EmptyOptimizer()))),
            output = Gamma(2.0, 1.0)
        )
    ]

    @test_rules [check_type_promotion = true] DeltaFn{g}((:in, k = 2), Marginalisation) [
        (
            input = (
                q_ins = FactorizedJoint((GammaShapeRate(2, 2), NormalMeanVariance())), m_in = NormalMeanVariance(1, 2), meta = DeltaMeta(method = CVI(1, 1, EmptyOptimizer()))
            ),
            output = NormalMeanVariance(-1.0, 2.0)
        ),
        (
            input = (q_ins = FactorizedJoint((NormalMeanVariance(), GammaShapeRate(2, 2))), m_in = GammaShapeRate(1, 1), meta = DeltaMeta(method = CVI(1, 1, EmptyOptimizer()))),
            output = Gamma(2.0, 1.0)
        )
    ]
end # testset
end # module

@testitem "rules:GammaShapeRate:α" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules, GammaShapeLikelihood

    @testset "Variational Message Passing: (q_out::Any, q_β::GammaDistributionsFamily)" begin
        @test_rules [check_type_promotion = true] GammaShapeRate(:α, Marginalisation) [
            (input = (q_out = GammaShapeRate(1.0, 1.0), q_β = GammaShapeRate(1.0, 1.0)), output = GammaShapeLikelihood(1.0, 2.0 * -0.5772156649015315)),
            (input = (q_out = PointMass(1.0), q_β = GammaShapeRate(1.0, 1.0)), output = GammaShapeLikelihood(1.0, -0.5772156649015315))
        ]
    end
end

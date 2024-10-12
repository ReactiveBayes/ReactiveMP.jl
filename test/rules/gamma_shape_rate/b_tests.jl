@testitem "rules:GammaShapeRate:β" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules, GammaShapeLikelihood

    @testset "Variational Message Passing: (q_out::Any, q_α::Any)" begin
        @test_rules [check_type_promotion = true] GammaShapeRate(:β, Marginalisation) [
            (input = (q_out = GammaShapeRate(1.0, 1.0), q_α = GammaShapeRate(1.0, 1.0)), output = GammaShapeRate(2.0, 1.0)),
            (input = (q_out = PointMass(1.0), q_α = GammaShapeRate(1.0, 1.0)), output = GammaShapeRate(2.0, 1.0)),
            (input = (q_out = GammaShapeScale(1.0, 1.0), q_α = PointMass(10.0)), output = GammaShapeRate(11.0, 1.0)),
            (input = (q_out = GammaShapeScale(1.0, 10.0), q_α = GammaShapeRate(1.0, 1.0)), output = GammaShapeRate(2, 10))
        ]
    end
end

@testitem "rules:GammaMixture:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
    import ReactiveMP: @test_rules
    # Tests for rule: GammaMixture(:switch, Marginalisation)
    @testset "Variational : (:switch)" begin
        @test_rules [check_type_promotion = false, atol = 1e-6] GammaMixture{2}(:switch, Marginalisation) [
            (
                input = (
                    q_out = GammaShapeRate(2.0, 1.0),
                    q_a = ManyOf((GammaShapeRate(1.0, 2.0), GammaShapeRate(2.0, 3.0))),
                    q_b = ManyOf((GammaShapeRate(3.0, 1.0), GammaShapeRate(4.0, 2.0)))
                ),
                output = Categorical([0.08088693183519022, 0.9191130681648099])
            )
        ]
    end
end

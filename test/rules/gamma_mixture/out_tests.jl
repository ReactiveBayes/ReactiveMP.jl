@testitem "rules:GammaMixture:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
    import ReactiveMP: @test_rules
    # Tests for rule: GammaMixture(:out, Marginalisation)
    @testset "Variational : (:out)" begin
        @test_rules [check_type_promotion = false] GammaMixture{2}(:out, Marginalisation) [
            (
                input = (q_switch = Categorical([0.7, 0.3]), q_a = ManyOf((PointMass(2.0), PointMass(3.0))), q_b = ManyOf((GammaShapeRate(1.0, 2.0), GammaShapeRate(2.0, 1.0)))),
                output = GammaShapeRate(
                    sum([0.7, 0.3] .* [mean(PointMass(2.0)), mean(PointMass(3.0))]), sum([0.7, 0.3] .* [mean(GammaShapeRate(1.0, 2.0)), mean(GammaShapeRate(2.0, 1.0))])
                )
            ),
            (
                input = (
                    q_switch = Categorical([0.5, 0.5]),
                    q_a = ManyOf((GammaShapeRate(2.0, 1.0), GammaShapeRate(3.0, 1.0))),
                    q_b = ManyOf((GammaShapeRate(1.0, 3.0), GammaShapeRate(1.0, 2.0)))
                ),
                output = GammaShapeRate(
                    sum([0.5, 0.5] .* [mean(GammaShapeRate(2.0, 1.0)), mean(GammaShapeRate(3.0, 1.0))]),
                    sum([0.5, 0.5] .* [mean(GammaShapeRate(1.0, 3.0)), mean(GammaShapeRate(1.0, 2.0))])
                )
            )
        ]
    end
end

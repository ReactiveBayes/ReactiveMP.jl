@testitem "rules:GammaMixture:b" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
    import ReactiveMP: @test_rules
    # Tests for rule: GammaMixture((:b, k), Marginalisation)
    @testset "Variational : (:b, k=1)" begin
        @test_rules [check_type_promotion = false] GammaMixture{2}((:b, k = 1), Marginalisation) [
            (
                input = (q_out = GammaShapeRate(2.0, 1.0), q_switch = Categorical([0.6, 0.4]), q_a = PointMass(1.0)),
                output = GammaShapeRate(1 + 0.6 * mean(PointMass(1.0)), 0.6 * mean(GammaShapeRate(2.0, 1.0)))
            ),
            ( # due to probvec(Bernoulli(0.8)) = (0.19999999999999996, 0.8) - and thus π_k ≈ 0.2 and not 0.8
                input = (q_out = GammaShapeRate(4.0, 2.0), q_switch = Bernoulli(0.8), q_a = GammaShapeRate(2.0, 3.0)),
                output = GammaShapeRate(1 + 0.2 * mean(GammaShapeRate(2.0, 3.0)), 0.2 * mean(GammaShapeRate(4.0, 2.0)))
            )
        ]
    end
end

@testitem "rules:MultinomialPolya:x" begin
    using ReactiveMP, BayesBase, Random, Distributions,ExponentialFamily
    import ReactiveMP: @test_rules

    @testset "Predictive Distribution: (q_N::PointMass, q_ψ::GaussianDistributionsFamily, meta::Union{Nothing, MultinomialPolyaMeta})" begin
        q_N = PointMass(3)
        q_ψ = MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0])
        metas = [nothing, MultinomialPolyaMeta(1, MersenneTwister(42))]
        
        # Expected values from logistic stick-breaking
        p_expected = [0.5, 0.25, 0.25]  # logistic(0) = 0.5, then 0.5*(1-0.5)=0.25, remainder 0.25
        dist_expected = Multinomial(3, p_expected)

        @test_rules [check_type_promotion = false] MultinomialPolya(:x, Marginalisation) [(
            input = (q_N = q_N, q_ψ = q_ψ, meta = metas[1]),
            output = dist_expected
        )]

        for meta in metas
            out = @call_rule MultinomialPolya(:x, Marginalisation) (q_N = q_N, q_ψ = q_ψ, meta = meta)
            @test out isa Multinomial
            @test ntrials(out) == 3
            @test probs(out) ≈ p_expected atol=1e-6
        end
    end
end

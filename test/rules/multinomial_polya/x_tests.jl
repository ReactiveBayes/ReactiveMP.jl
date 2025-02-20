@testitem "rules:MultinomialPolya:x" begin
    using ReactiveMP, BayesBase, Random, Distributions, ExponentialFamily
    import ReactiveMP: @test_rules

    @testset "Predictive Distribution: (q_N::PointMass, q_ψ::GaussianDistributionsFamily)" begin
        q_N = PointMass(3)
        q_ψ = MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0])

        # Expected values from logistic stick-breaking
        p_expected = [0.5, 0.25, 0.25]  # logistic(0) = 0.5, then 0.5*(1-0.5)=0.25, remainder 0.25
        dist_expected = Multinomial(3, p_expected)

        @test_rules [check_type_promotion = false] MultinomialPolya(:x, Marginalisation) [(input = (q_N = q_N, q_ψ = q_ψ), output = dist_expected)]
    end

    @testset "Predictive Distribution: (q_N::Poisson, q_ψ::GaussianDistributionsFamily)" begin
        q_Ns = [Poisson(3), Binomial(5, 0.5), Categorical([0.01, 0.01, 0.98])]
        q_ψ = MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0])

        # Expected values from logistic stick-breaking
        p_expected = [0.5, 0.25, 0.25]  # logistic(0) = 0.5, then 0.5*(1-0.5)=0.25, remainder 0.25
        dist_expected = Multinomial(3, p_expected)
        for q_N in q_Ns
            @test_rules [check_type_promotion = false] MultinomialPolya(:x, Marginalisation) [(input = (q_N = q_N, q_ψ = q_ψ), output = dist_expected)]
        end
    end
end

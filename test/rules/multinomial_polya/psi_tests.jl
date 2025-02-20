@testitem "rules:MultinomialPolya:psi" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, PolyaGammaHybridSamplers, StableRNGs

    import ReactiveMP: @test_rules, weightedmean
    import LinearAlgebra: Diagonal, diag

    @testset "Expectation Propagation: (q_x::PointMass, q_N::PointMass, m_ψ::GaussianDistributionsFamily)" begin
        rng = StableRNG(42)
        q_x = PointMass([0, 1, 2])
        q_N = PointMass(3)
        m_ψ = MvNormalWeightedMeanPrecision(zeros(2), diageye(2))

        # Expected values based on the rule calculations
        # η = η_ψ + (x[1:K-1] - Nks/2)
        # Λ = Λ_ψ + Diagonal(ω)
        η_expected = [-1.5, -0.5]
        Λ_expected = Diagonal([0.75, 0.75])

        @test_rules [check_type_promotion = false] MultinomialPolya(:ψ, Marginalisation) [(
            input = (q_x = q_x, q_N = q_N, m_ψ = m_ψ), output = MvGaussianWeightedMeanPrecision(η_expected, Λ_expected)
        )]
    end

    @testset "Expectation Propagation: (q_x::PointMass, q_N::Poisson, m_ψ::GaussianDistributionsFamily)" begin
        q_x = PointMass([0, 1, 2])
        q_Ns = [Poisson(3), Binomial(5, 0.5), Categorical([0.01, 0.01, 0.98])]
        m_ψ = MvNormalWeightedMeanPrecision(zeros(2), diageye(2))

        η_expected = [-1.5, -0.5]
        Λ_expected = Diagonal([0.75, 0.75])

        for q_N in q_Ns
            @test_rules [check_type_promotion = false] MultinomialPolya(:ψ, Marginalisation) [(
                input = (q_x = q_x, q_N = q_N, m_ψ = m_ψ), output = MvGaussianWeightedMeanPrecision(η_expected, Λ_expected)
            )]
        end
    end
end

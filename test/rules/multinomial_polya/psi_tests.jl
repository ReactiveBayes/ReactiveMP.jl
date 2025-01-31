@testitem "rules:MultinomialPolya:psi" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, PolyaGammaHybridSamplers

    import ReactiveMP: @test_rules, weightedmean
    import LinearAlgebra: Diagonal,diag

    @testset "Expectation Propagation: (q_x::PointMass, q_N::PointMass, m_ψ::GaussianDistributionsFamily, meta::Union{Nothing, MultinomialPolyaMeta})" begin
        q_x = PointMass([0, 1, 2])
        q_N = PointMass(3)
        m_ψ = MvNormalWeightedMeanPrecision(zeros(2), diageye(2))
        metas = [nothing, MultinomialPolyaMeta(1, MersenneTwister(10)), MultinomialPolyaMeta(100, MersenneTwister(42))]

        # Expected values based on the rule calculations
        # η = η_ψ + (x[1:K-1] - Nks/2)
        # Λ = Λ_ψ + Diagonal(ω)
        η_expected = [-1.5, -0.5]
        Λ_expected = Diagonal([0.75, 0.75])

        @test_rules [check_type_promotion = false] MultinomialPolya(:ψ, Marginalisation) [(
            input = (q_x = q_x, q_N = q_N, m_ψ = m_ψ, meta = metas[1]), 
            output = MvGaussianWeightedMeanPrecision(η_expected, Λ_expected)
        )]

        for meta in metas
            out = @call_rule MultinomialPolya(:ψ, Marginalisation) (q_x = q_x, q_N = q_N, m_ψ = m_ψ, meta = meta)
            @test weightedmean(out) ≈ η_expected rtol = 1e-8
            @test precision(out) ≈ Λ_expected atol = 1e-1
        end
    end
end

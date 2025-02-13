
@testitem "rules:BinomialPolya:beta" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, PolyaGammaHybridSamplers

    import ReactiveMP: @test_rules, weightedmean
    import LinearAlgebra: diag

    @testset "Expectation Propagation: (q_y::PointMass, q_x::PointMass, q_n::PointMass, m_β::GaussianDistributionsFamily, meta::Union{Nothing, BinomialPolyaMeta})" begin
        q_y = PointMass(3)
        q_x = PointMass([0.1, 0.2])
        q_n = PointMass(5)
        m_β = MvNormalWeightedMeanPrecision(zeros(2), diageye(2))
        metas = [nothing, BinomialPolyaMeta(1, MersenneTwister(10)), BinomialPolyaMeta(10, MersenneTwister(42))]

        ## Hand-calculated values. If meta is not nothing, the values will be a bit different.
        Λ = [0.0125 0.025; 0.025 0.05]
        xi = [0.05, 0.1]

        @test_rules [check_type_promotion = true] BinomialPolya(:β, Marginalisation) [(
            input = (q_y = q_y, q_x = q_x, q_n = q_n, m_β = m_β, meta = metas[1]), output = MvNormalWeightedMeanPrecision(xi, Λ)
        )]

        for meta in metas
            out = @call_rule BinomialPolya(:β, Marginalisation) (q_y = q_y, q_x = q_x, q_n = q_n, m_β = m_β, meta = meta)
            @test weightedmean(out) ≈ xi rtol = 1e-8
            @test diag(precision(out)) ≈ diag(Λ) atol = 1e-2
        end
    end

    @testset "Expectation Propagation: Univariate (q_y::PointMass, q_x::PointMass, q_n::PointMass, m_β::NormalDistributionsFamily)" begin
        q_y = PointMass(3)
        q_x = PointMass(0.1)  # 1-dimensional x
        q_n = PointMass(5)
        m_β = NormalWeightedMeanPrecision(0.0, 1.0)  # Univariate normal
        metas = [nothing, BinomialPolyaMeta(1, MersenneTwister(10))]

        Λ = 0.0125
        ξ = 0.05

        @test_rules [check_type_promotion = true] BinomialPolya(:β, Marginalisation) [(
            input = (q_y = q_y, q_x = q_x, q_n = q_n, m_β = m_β, meta = metas[1]), output = NormalWeightedMeanPrecision(ξ, Λ)
        )]

        for meta in metas
            out = @call_rule BinomialPolya(:β, Marginalisation) (q_y = q_y, q_x = q_x, q_n = q_n, m_β = m_β, meta = meta)
            @test weightedmean(out) ≈ ξ rtol = 1e-8
            @test precision(out) ≈ Λ atol = 1e-2
        end
    end
end

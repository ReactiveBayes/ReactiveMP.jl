
@testitem "marginalrules:SoftDot" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_marginalrules

    @testset "y_x: (m_y::UnivariateNormalDistributionsFamily, m_x::UnivariateNormalDistributionsFamily, q_θ::UnivariateNormalDistributionsFamily, q_γ::Any)" begin
        @test_marginalrules [check_type_promotion = true] SoftDot(:y_x) [(
            input = (m_y = NormalMeanPrecision(0.0, 1.0), m_x = NormalMeanPrecision(0.0, 1.0), q_θ = NormalMeanPrecision(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0)),
            output = MvNormalWeightedMeanPrecision(zeros(2), [2.0 -1.0; -1.0 3.0])
        )]
    end

    @testset "y_x: (m_y::UnivariateNormalDistributionsFamily), m_x::MultivariateNormalDistributionsFamily, q_θ::MultivariateNormalDistributionsFamily, q_γ::Any)" begin
        order = 2
        @test_marginalrules [check_type_promotion = true] SoftDot(:y_x) [(
            input = (
                m_y = NormalMeanPrecision(1.0, 1.0),
                m_x = MvNormalMeanCovariance(ones(order), diageye(order)),
                q_θ = MvNormalMeanCovariance(ones(order), diageye(order)),
                q_γ = GammaShapeRate(1.0, 1.0)
            ),
            output = MvNormalWeightedMeanPrecision(ones(3), [2.0 -1.0 -1.0; -1.0 3.0 1.0; -1.0 1.0 3.0])
        )]
    end
end

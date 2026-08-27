@testitem "rules:Bilinear:marginals" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_marginalrules

    @testset "out_in: (m_out::UnivariateNormalDistributionsFamily, m_in::UnivariateNormalDistributionsFamily, q_a::PointMass)" begin
        @test (@call_marginalrule Bilinear(:out_in) (
            m_out = NormalWeightedMeanPrecision(1.0, 2.0),
            m_in = NormalWeightedMeanPrecision(2.0, 3.0),
            q_a = PointMass(1.0),
        )) ≈ MvNormalWeightedMeanPrecision([1.0, 2.0], [2.0 -1.0; -1.0 3.0])
    end
end

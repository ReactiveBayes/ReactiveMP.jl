@testitem "marginalrules:Binomial" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_marginalrules

    @testset "n_k_p: (m_n::Any, m_k::Any, m_p::Any)" begin

        @test_marginalrules [check_type_promotion = true] Binomial(:n_k_p) [
            (input = (m_n = PointMass(2), m_k = PointMass(10), m_p = PointMass(0.5)), output = (n = Beta(1.0, 3.0), k = PointMass(1.0), p = PointMass(2.0))),
        ]
    end
end
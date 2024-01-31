
@testitem "rules:Poisson:l" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_out::PointMass)" begin
        @test_rules [check_type_promotion = true] Poisson(:l, Marginalisation) [
            (input = (m_out = PointMass(1.0),), output = Gamma(2.0, 1.0)), (input = (m_out = PointMass(0.2),), output = Gamma(1.2, 1.0))
        ]
    end

    @testset "Variational Message Passing: (q_out::PointMass)" begin
        @test_rules [check_type_promotion = true] Poisson(:l, Marginalisation) [
            (input = (q_out = Poisson(1.0),), output = Gamma(2.0, 1.0)), (input = (q_out = Poisson(0.3),), output = Gamma(1.3, 1.0))
        ]
    end
end

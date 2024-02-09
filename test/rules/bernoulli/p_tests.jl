
@testitem "rules:Bernoulli:p" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_out::PointMass)" begin
        @test_rules [check_type_promotion = true] Bernoulli(:p, Marginalisation) [
            (input = (m_out = PointMass(1.0),), output = Beta(2.0, 1.0)), (input = (m_out = PointMass(0.2),), output = Beta(12 / 10, 9 / 5))
        ]
    end

    @testset "Variational Message Passing: (q_out::Bernoulli)" begin
        @test_rules [check_type_promotion = true] Bernoulli(:p, Marginalisation) [
            (input = (q_out = Bernoulli(1.0),), output = Beta(2.0, 1.0)), (input = (q_out = Bernoulli(0.3),), output = Beta(13 / 10, 17 / 10))
        ]
    end

    @testset "Variational Message Passing: (q_out::DiscreteNonParametric)" begin
        # `check_type_promotion = false` here is because apparently 
        # BigFloat(0.7) + BigFloat(0.3) != BigFloat(1.0)
        @test_rules [check_type_promotion = false] Bernoulli(:p, Marginalisation) [
            (input = (q_out = Categorical([0.0, 1.0]),), output = Beta(2.0, 1.0)), (input = (q_out = Categorical([0.7, 0.3]),), output = Beta(13 / 10, 17 / 10))
        ]
    end
end

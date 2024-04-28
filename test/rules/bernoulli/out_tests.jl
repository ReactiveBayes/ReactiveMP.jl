
@testitem "rules:Bernoulli:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_p::PointMass)" begin
        @test_rules [check_type_promotion = true] Bernoulli(:out, Marginalisation) [
            (input = (m_p = PointMass(1.0),), output = Bernoulli(1.0)), (input = (m_p = PointMass(0.2),), output = Bernoulli(0.2))
        ]
    end

    @testset "Variational Message Passing: (q_p::PointMass)" begin
        @test_rules [check_type_promotion = true] Bernoulli(:out, Marginalisation) [
            (input = (q_p = PointMass(1.0),), output = Bernoulli(1.0)), (input = (q_p = PointMass(0.3),), output = Bernoulli(0.3))
        ]
    end

    @testset "Variational Message Passing: (q_p::Beta)" begin
        @test_rules [check_type_promotion = true] Bernoulli(:out, Marginalisation) [
            (input = (q_p = Beta(1.0, 1.0),), output = Bernoulli(0.5)), (input = (q_p = Beta(0.2, 0.2),), output = Bernoulli(0.5))
        ]
    end
end

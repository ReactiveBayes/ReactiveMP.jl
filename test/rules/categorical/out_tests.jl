
@testitem "rules:Categorical:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_p::PointMass)" begin
        @test_rules [check_type_promotion = false] Categorical(:out, Marginalisation) [
            (input = (m_p = PointMass([0.0, 1.0]),), output = Categorical([0.0, 1.0])), (input = (m_p = PointMass([0.8, 0.2]),), output = Categorical([0.8, 0.2]))
        ]
    end

    @testset "Variational Message Passing: (q_p::PointMass)" begin
        @test_rules [check_type_promotion = false] Categorical(:out, Marginalisation) [
            (input = (q_p = PointMass([0.0, 1.0]),), output = Categorical([0.0, 1.0])), (input = (q_p = PointMass([0.7, 0.3]),), output = Categorical([0.7, 0.3]))
        ]
    end

    @testset "Variational Message Passing: (q_p::Dirichlet)" begin
        @test_rules [check_type_promotion = false] Categorical(:out, Marginalisation) [
            (input = (q_p = Dirichlet([1.0, 1.0]),), output = Categorical([0.5, 0.5])), (input = (q_p = Dirichlet([0.2, 0.2]),), output = Categorical([0.5, 0.5]))
        ]
    end
end

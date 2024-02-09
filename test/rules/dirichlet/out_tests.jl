
@testitem "rules:Dirichlet:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_a::PointMass)" begin
        @test_rules [check_type_promotion = true] Dirichlet(:out, Marginalisation) [
            (input = (m_a = PointMass([0.2, 1.0]),), output = Dirichlet([0.2, 1.0])),
            (input = (m_a = PointMass([2.0, 0.5]),), output = Dirichlet([2.0, 0.5])),
            (input = (m_a = PointMass([3.0, 1.0]),), output = Dirichlet([3.0, 1.0]))
        ]
    end

    @testset "Variational Message Passing: (q_a::PointMass)" begin
        @test_rules [check_type_promotion = true] Dirichlet(:out, Marginalisation) [
            (input = (q_a = PointMass([0.2, 1.0]),), output = Dirichlet([0.2, 1.0])),
            (input = (q_a = PointMass([2.0, 0.5]),), output = Dirichlet([2.0, 0.5])),
            (input = (q_a = PointMass([3.0, 1.0]),), output = Dirichlet([3.0, 1.0]))
        ]
    end
end

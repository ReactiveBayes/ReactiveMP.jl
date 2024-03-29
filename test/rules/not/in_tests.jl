
@testitem "rules:NOT:in" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_in::Bernoulli)" begin
        @test_rules [check_type_promotion = true] NOT(:in, Marginalisation) [
            (input = (m_out = Bernoulli(0.6),), output = Bernoulli(0.4)), (input = (m_out = Bernoulli(0.3),), output = Bernoulli(0.7))
        ]
    end
end

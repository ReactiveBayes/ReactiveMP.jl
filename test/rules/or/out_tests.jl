
@testitem "rules:OR:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_in1::Bernoulli, m_in2::Bernoulli)" begin
        @test_rules [check_type_promotion = true] OR(:out, Marginalisation) [
            (input = (m_in1 = Bernoulli(0.5), m_in2 = Bernoulli(0.5)), output = Bernoulli(0.75)),
            (input = (m_in1 = Bernoulli(0.3), m_in2 = Bernoulli(0.4)), output = Bernoulli(0.58))
        ]
    end
end

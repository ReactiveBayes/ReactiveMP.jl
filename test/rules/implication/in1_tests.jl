
@testitem "rules:IMPLY:in1" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_out::Bernoulli, m_in2::Bernoulli)" begin
        @test_rules [check_type_promotion = true] IMPLY(:in1, Marginalisation) [
            (input = (m_out = Bernoulli(0.6), m_in2 = Bernoulli(0.5)), output = Bernoulli(0.5 / 1.1)),
            (input = (m_out = Bernoulli(0.2), m_in2 = Bernoulli(0.5)), output = Bernoulli(0.5 / 0.7))
        ]
    end
end


@testitem "rules:IMPLY:in2" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_out::Bernoulli, m_in1::Bernoulli)" begin
        @test_rules [check_type_promotion = true] IMPLY(:in2, Marginalisation) [
            (input = (m_out = Bernoulli(0.6), m_in1 = Bernoulli(0.5)), output = Bernoulli(0.6 / 1.1)),
            (input = (m_out = Bernoulli(0.3), m_in1 = Bernoulli(0.4)), output = Bernoulli(0.3 / 0.76))
        ]
    end
end

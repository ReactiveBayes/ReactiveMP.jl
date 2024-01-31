
@testitem "rules:NOT:marginals" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules, @test_marginalrules

    @testset ":in (m_out::Bernoulli, m_in::Bernoulli)" begin
        @test_marginalrules [check_type_promotion = true] NOT(:in) [
            (input = (m_out = Bernoulli(0.4), m_in = Bernoulli(0.5)), output = Bernoulli(0.6)),
            (input = (m_out = Bernoulli(0.2), m_in = Bernoulli(0.8)), output = Bernoulli(0.64 / (0.68)))
        ]
    end
end

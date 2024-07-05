@testitem "Generic nodefunction fallback rule" begin
    using Distributions, BayesBase

    struct MyBeta{A, B} <: ContinuousUnivariateDistribution
        a::A
        b::B
    end

    BayesBase.logpdf(d::MyBeta, x) = logpdf(Beta(d.a, d.b), x)
    BayesBase.insupport(d::MyBeta, x::Real) = true

    @node MyBeta Stochastic [out, a, b]

    m_a = Beta(3, 7)
    m_b = Beta(7, 3)
    m_out = Beta(314, 42)

    @test_throws ReactiveMP.RuleMethodError @call_rule MyBeta(:out, Marginalisation) (m_a = m_a, m_b = m_b)

    for f in (mean, mode)
        message_out = @call_rule [fallback = NodeFunctionRuleFallback(f)] MyBeta(:out, Marginalisation) (m_a = m_a, m_b = m_b)
        message_a = @call_rule [fallback = NodeFunctionRuleFallback(f)] MyBeta(:a, Marginalisation) (m_out = m_out, m_b = m_b)
        message_b = @call_rule [fallback = NodeFunctionRuleFallback(f)] MyBeta(:b, Marginalisation) (m_out = m_out, m_a = m_a)

        for p in (0.1:0.1:0.9)
            @test logpdf(message_out, p) ≈ logpdf(Beta(f(m_a), f(m_b)), p)
            @test logpdf(message_a, p) ≈ logpdf(Beta(p, f(m_b)), f(m_out))
            @test logpdf(message_b, p) ≈ logpdf(Beta(f(m_a), p), f(m_out))
        end

        # mean-field
        message_out = @call_rule [fallback = NodeFunctionRuleFallback(f)] MyBeta(:out, Marginalisation) (q_a = m_a, q_b = m_b)
        message_a = @call_rule [fallback = NodeFunctionRuleFallback(f)] MyBeta(:a, Marginalisation) (q_out = m_out, q_b = m_b)
        message_b = @call_rule [fallback = NodeFunctionRuleFallback(f)] MyBeta(:b, Marginalisation) (q_out = m_out, q_a = m_a)

        for p in (0.1:0.1:0.9)
            @test logpdf(message_out, p) ≈ logpdf(Beta(f(m_a), f(m_b)), p)
            @test logpdf(message_a, p) ≈ logpdf(Beta(p, f(m_b)), f(m_out))
            @test logpdf(message_b, p) ≈ logpdf(Beta(f(m_a), p), f(m_out))
        end
    end
end
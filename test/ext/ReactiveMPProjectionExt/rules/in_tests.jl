@testitem "Basic check for in rule" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    meta = DeltaMeta(method = CVIProjection(), inverse = nothing)
    m_in_incoming = NormalMeanVariance(0, 1)
    m_in_outbound_real = NormalMeanVariance(0, 1)
    q_ins = FactorizedJoint((prod(GenericProd(), m_in_incoming, m_in_outbound_real),))
    m_in_outbound_approximated = @call_rule DeltaFn{identity}((:in, k = 1), Marginalisation) (q_ins = q_ins, m_in = m_in_incoming, meta = meta)

    for x in randn(100)
        @test logpdf(m_in_outbound_approximated, x) + logpdf(m_in_incoming, x) ≈ logpdf(q_ins[1], x)
    end
end

@testitem "Basic check for in rule" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    meta = DeltaMeta(method = CVIProjection(), inverse = nothing)
    m_in_incoming = NormalMeanVariance(0, 1)
    m_in_outbound_real = NormalMeanVariance(0, 1)
    q_ins = FactorizedJoint((prod(GenericProd(), m_in_incoming, m_in_outbound_real),))

    m_in_outbound_approximated = @call_rule DeltaFn{identity}((:in, k = 1), Marginalisation) (q_ins = q_ins, m_in = m_in_incoming, meta = meta)
    @test prod(GenericProd(), m_in_outbound_approximated, m_in_incoming) ≈ q_ins[1]
end

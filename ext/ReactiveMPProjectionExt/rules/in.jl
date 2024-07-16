
@rule DeltaFn((:in, k), Marginalisation) (q_ins::FactorizedJoint, m_in::Any, meta::DeltaMeta{M}) where {M <: CVIProjection} = begin
    q_ins_k = component(q_ins, k)
    # return z -> logpdf(q_ins_k, z) - logpdf(m_in, z)
    return DivisionOf(q_ins_k, m_in)
end

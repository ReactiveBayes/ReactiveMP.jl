
@rule DeltaFn((:in, k), Marginalisation) (q_ins::FactorizedJoint, m_in::Any, meta::DeltaMeta{M}) where {M <: CVIProjection} = begin
    q_ins_k = component(q_ins, k)
    return DivisionOf(q_ins_k_ef, m_in_ef)
end

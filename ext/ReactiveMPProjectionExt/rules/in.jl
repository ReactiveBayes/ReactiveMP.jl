
@rule DeltaFn((:in, k), Marginalisation) (q_ins::FactorizedJoint, m_in::Any, meta::DeltaMeta{M}) where {M <: CVIProjection} = begin
    q_ins_k = component(q_ins, k)
    # Check if both the marginal and the message are from the same exponential family
    if !(ExponentialFamily.exponential_family_typetag(q_ins_k) === ExponentialFamily.exponential_family_typetag(m_in))
        error("Cannot compute a message with `CVIProjection`. The marginal and the inbound message are not of the same exponential family.")
    end
    q_ins_k_ef = convert(ExponentialFamilyDistribution, q_ins_k)
    m_in_ef = convert(ExponentialFamilyDistribution, m_in)
    if !(getconditioner(q_ins_k_ef) == getconditioner(m_in_ef))
        error("Cannot compute a message with `CVIProjection`. Conditioners for the marginal and for the inbound message are different.")
    end
    return DivisionOf(q_ins_k_ef, m_in_ef)
end

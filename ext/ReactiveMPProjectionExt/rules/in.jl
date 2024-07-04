
@rule DeltaFn((:in, k), Marginalisation) (q_ins::FactorizedJoint, m_in::Any, meta::DeltaMeta{M}) where {M <: CVIProjection} = begin
    q_ins_k = component(q_ins, k)
    # Check if both the marginal and the message are from the same exponential family
    if ExponentialFamily.exponential_family_typetag(q_ins_k) === ExponentialFamily.exponential_family_typetag(m_in)
        q_ins_k_ef = convert(ExponentialFamilyDistribution, q_ins_k)
        m_in_ef = convert(ExponentialFamilyDistribution, m_in)
        if getconditioner(q_ins_k_ef) !== getconditioner(m_in_ef)
            error("Cannot compute a message with `CVI`. Conditioners for the marginal and for the inbound message are different.")
        end
        T = ExponentialFamily.exponential_family_typetag(q_ins_k)
        η = getnaturalparameters(q_ins_k_ef) - getnaturalparameters(m_in_ef)
        c = getconditioner(q_ins_k_ef)
        # TODO: `nothing` here bypasses validity check for the natural parameters `η`
        # This is not ideal, we should investigate that further and maybe throw an error if the resulting natural parameter are not valid
        result = ExponentialFamilyDistribution(T, η, c, nothing)
        return convert(Distribution, result)
    end
end

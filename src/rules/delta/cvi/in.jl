using Random
import Distributions: Distribution

@rule DeltaFn((:in, k), Marginalisation) (q_ins::FactorizedJoint, m_in::Any, meta::DeltaMeta{M}) where {M <: CVIApproximation} = begin
    return convert(Distribution, naturalparams(q_ins[k]) - naturalparams(m_in))
end

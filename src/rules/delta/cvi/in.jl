using Random
import Distributions: Distribution

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::FactorizedJoint, m_in::Any, meta::DeltaMeta{M}) where {f, M <: CVIApproximation} = begin
    return convert(Distribution, naturalparams(q_ins[k]) - naturalparams(m_in))
end

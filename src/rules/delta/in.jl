using Random
import Distributions: Distribution

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::FactorProduct, m_in::Any, meta::CVIApproximation) where {f} = begin
    return convert(Distribution, naturalparams(q_ins[k]) - naturalparams(m_in))
end

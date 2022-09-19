using Random

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::FactorProduct, m_in::Any, meta::CVIApproximation) where {f} = begin
    return standardDist(naturalParams(q_ins[k]) - naturalParams(m_in))
end

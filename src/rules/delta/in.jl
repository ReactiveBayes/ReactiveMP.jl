using Random

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::FactorProduct, m_in::Any, meta::CVIApproximation) where {f} = begin
    return standardDist(naturalparams(q_ins[k]) - naturalparams(m_in))
end

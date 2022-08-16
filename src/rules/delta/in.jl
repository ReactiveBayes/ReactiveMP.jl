using Random

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::SamplingApproximation) where {f} = begin
    return ContinuousUnivariateLogPdf((x) -> logpdf(m_in, f(x)))
end

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::FactorProduct, m_in::Any, meta::CVIApproximation) where {f} = begin
    return standardDist(naturalParams(q_ins[k]) - naturalParams(m_in))
end

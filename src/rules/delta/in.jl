using Random

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::SamplingApproximation) where {f} = begin
    return ContinuousUnivariateLogPdf((x) -> logpdf(m_in, f(x)))
end

@rule DeltaFn{f}((:in, k), Marginalisation) (
    q_ins::Any,
    m_in::Any,
    meta::LinearApproximationKnownInverse
) where {f} = begin
    return NormalMeanVariance(0, 1)
end

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::FactorProduct, m_in::Any, meta::CVIApproximation) where {f} = begin
    return q_ins[k]
end

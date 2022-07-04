using Random 

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::SamplingApproximation) where {f} = begin
    return ContinuousUnivariateLogPdf((x) -> logpdf(m_in, f(x)))
end

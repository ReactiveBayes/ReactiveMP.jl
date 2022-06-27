@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::SamplingApproximation) = begin
    return NormalMeanPrecision(0.0, 1.0)
end

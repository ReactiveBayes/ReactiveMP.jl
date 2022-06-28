@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::SamplingApproximation) where {f} = begin
    return NormalMeanPrecision(0.0, 1.0)
end

@rule DeltaFn{f}((:in, k), Marginalisation) (
    q_ins::Any,
    m_in::Any,
    meta::LinearApproximationKnownInverse
) where {f} = begin
    return NormalMeanVariance(0, 1)
end

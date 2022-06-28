@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::SamplingApproximation) where {f} = begin
    return NormalMeanPrecision(0.0, 1.0)
end

@rule DeltaFn{f}((:in, k), Marginalisation) (
    m_out::UnivariateDistribution,
    meta::LinearApproximationKnownInverse
) where {f} = begin
    (m_out, V_out) = mean(m_out), var(m_out)
    (A, b) = localLinearizationSingleIn(meta.F_inv, m_out)
    m = A * m_out + b
    V = A * V_out * A'
    return NormalMeanVariance(m, V)
end

@rule DeltaFn{f}(:out, Marginalisation) (m_out::Any, m_ins::NTuple{N, Any}, meta::DeltaExtended{T}) where {f, N, T} =
    begin
        (ms_fw_in, Vs_fw_in) = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
        (A, b) = localLinearization(f, ms_fw_in)
        (m_fw_in, V_fw_in, _) = concatenateGaussianMV(ms_fw_in, Vs_fw_in)
        m = A * m_fw_in + b
        V = A * V_fw_in * A'

        F = size(m, 1) == 1 ? Univariate : Multivariate
        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

@rule DeltaFn{f}(:out, Marginalisation) (m_ins::NTuple{1, NormalDistributionsFamily}, meta::DeltaExtended{T}) where {f, T} = begin
    μ_in, Σ_in = mean_cov(first(m_ins))
    (A, b) = localLinearization(f, μ_in)
    m = A * μ_in + b
    V = A * Σ_in * A'
    F = size(m, 1) == 1 ? Univariate : Multivariate
    return convert(promote_variate_type(F, NormalMeanVariance), m, V)
end

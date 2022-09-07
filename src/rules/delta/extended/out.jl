# single input
@rule DeltaFn{f}(:out, Marginalisation) (
    m_ins::NTuple{1, NormalDistributionsFamily},
    meta::DeltaExtended{T}
) where {f, T} = begin
    μ_in, Σ_in = mean_cov(first(m_ins))
    (A, b) = localLinearizationSingleIn(f, μ_in)
    m = A * μ_in + b
    V = A * Σ_in * A'
    F = size(m, 1) == 1 ? Univariate : Multivariate
    return convert(promote_variate_type(F, NormalMeanVariance), m, V)
end

# multiple input; this should be called
@rule DeltaFn{f}(:out, Marginalisation) (
    m_ins::NTuple{N, NormalDistributionsFamily},
    meta::DeltaExtended{T}
) where {f, N, T} =
    begin
        (μs_in, Σs_in) = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
        (A, b) = localLinearizationMultiIn(f, μs_in)
        (m_fw_in, V_fw_in, _) = concatenateGaussianMV(μs_in, Σs_in)
        m = A * m_fw_in + b
        V = A * V_fw_in * A'
        F = size(m, 1) == 1 ? Univariate : Multivariate
        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

# Why this method is being called for forward message?
@rule DeltaFn{f}(:out, Marginalisation) (
    m_out::NormalDistributionsFamily,
    m_ins::NTuple{N, NormalDistributionsFamily},
    meta::DeltaExtended{T}
) where {f, N, T} =
    begin
        (μs_in, Σs_in) = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
        (A, b) = localLinearizationMultiIn(f, μs_in)
        (μ_in, Σ_in, _) = concatenateGaussianMV(μs_in, Σs_in)

        m = A * μ_in + b
        V = A * Σ_in * A'

        F = size(m, 1) == 1 ? Univariate : Multivariate
        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

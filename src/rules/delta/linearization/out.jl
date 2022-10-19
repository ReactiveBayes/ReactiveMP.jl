# single input
@rule DeltaFn{f}(:out, Marginalisation) (m_ins::ManyOf{1, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {f, M <: Linearization} = begin
    μ_in, Σ_in = mean_cov(first(m_ins))
    (A, b) = localLinearizationSingleIn(f, μ_in)
    m = A * μ_in + b
    V = A * Σ_in * A'
    return convert(promote_variate_type(variate_form(m), NormalMeanVariance), m, V)
end

# multiple input; this should be called
@rule DeltaFn{f}(:out, Marginalisation) (m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {f, N, M <: Linearization} = begin
    (μs_in, Σs_in) = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
    (A, b) = localLinearizationMultiIn(f, μs_in)
    (m_fw_in, V_fw_in) = mean_cov(convert(JointNormal, μs_in, Σs_in))
    m = A * m_fw_in + b
    V = A * V_fw_in * A'
    return convert(promote_variate_type(variate_form(m), NormalMeanVariance), m, V)
end

@rule DeltaFn{f}(:out, Marginalisation) (
    m_out::NormalDistributionsFamily,
    m_ins::ManyOf{N, NormalDistributionsFamily},
    meta::DeltaMeta{M}
) where {f, N, M <: Linearization} = begin
    (μs_in, Σs_in) = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
    (A, b) = localLinearizationMultiIn(f, μs_in)
    (μ_in, Σ_in) = mean_cov(convert(JointNormal, μs_in, Σs_in))
    m = A * μ_in + b
    V = A * Σ_in * A'
    return convert(promote_variate_type(variate_form(m), NormalMeanVariance), m, V)
end

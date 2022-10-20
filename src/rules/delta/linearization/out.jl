

@rule DeltaFn{f}(:out, Marginalisation) (m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {f, N, M <: Linearization} = begin
    (μs_in, Σs_in) = mean_cov.(m_ins...) # Returns arrays with individual means and covariances
    (A, b) = approximate(f, μs_in)
    (m_fw_in, V_fw_in) = mean_cov(convert(JointNormal, μs_in, Σs_in))
    m = A * m_fw_in + b
    V = A * V_fw_in * A'
    return convert(promote_variate_type(variate_form(m), NormalMeanVariance), m, V)
end

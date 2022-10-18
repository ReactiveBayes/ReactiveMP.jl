# single input
@rule DeltaFn{f}(:out, Marginalisation) (m_ins::ManyOf{1, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {f, M <: Unscented} = begin
    (μ_fw_in1, Σ_fw_in1) = mean_cov(first(m_ins))
    (μ_tilde, Σ_tilde, _) = unscented_statistics(getmethod(meta), μ_fw_in1, Σ_fw_in1, f)
    return convert(promote_variate_type(variate_form(μ_tilde), NormalMeanVariance), μ_tilde, Σ_tilde)
end

# multiple input; this should be called
@rule DeltaFn{f}(:out, Marginalisation) (m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {f, N, M <: Unscented} = begin
    (μs_in, Σs_in)        = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
    (μ_tilde, Σ_tilde, _) = unscented_statistics(getmethod(meta), μs_in, Σs_in, f)
    return convert(promote_variate_type(variate_form(μ_tilde), NormalMeanVariance), μ_tilde, Σ_tilde)
end

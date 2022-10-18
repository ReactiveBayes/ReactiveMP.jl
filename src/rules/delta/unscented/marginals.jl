
# most of routines are ported from ForneyLab.jl

@marginalrule DeltaFn{f}(:ins) (m_out::NormalDistributionsFamily, m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {f, N, M <: Unscented} = begin
    # Approximate joint inbounds
    (μs_fw_in, Σs_fw_in) = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
    (μ_tilde, Σ_tilde, C_tilde) = unscented_statistics(getmethod(meta), μs_fw_in, Σs_fw_in, f)

    # RTS smoother
    joint              = convert(JointNormal, μs_fw_in, Σs_fw_in)
    (μ_fw_in, Σ_fw_in) = mean_cov(joint) 
    ds                 = dimensionalities(joint)

    (μ_bw_out, Σ_bw_out) = mean_cov(m_out)
    (μ_in, Σ_in) = smoothRTS(μ_tilde, Σ_tilde, C_tilde, μ_fw_in, Σ_fw_in, μ_bw_out, Σ_bw_out)
    return JointNormal(MvNormalMeanCovariance(μ_in, Σ_in), ds)
end

@marginalrule DeltaFn{f}(:ins) (m_out::NormalDistributionsFamily, m_ins::ManyOf{1, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {f, M <: Unscented} = begin
    # Approximate joint inbounds
    μ_fw_in, Σ_fw_in = collectStatistics(m_ins...)

    (μ_tilde, Σ_tilde, C_tilde) = unscented_statistics(getmethod(meta), μ_fw_in, Σ_fw_in, f)
    (μ_bw_out, Σ_bw_out) = mean_cov(m_out)

    (μ_in, Σ_in) = smoothRTS(μ_tilde, Σ_tilde, C_tilde, μ_fw_in, Σ_fw_in, μ_bw_out, Σ_bw_out)

    dist = convert(promote_variate_type(variate_form(μ_in), NormalMeanVariance), μ_in, Σ_in)
    ds   = [ (length(μ_in),) ]

    return JointNormal(dist, ds)
end

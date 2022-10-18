# most of the routines are ported directly from ForneyLab.jl

@marginalrule DeltaFn{f}(:ins) (m_out::NormalDistributionsFamily, m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {f, N, M <: Linearization} = begin
    # Approximate joint inbounds
    (μs_fw_in, Σs_fw_in) = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
    (A, b) = localLinearizationMultiIn(f, μs_fw_in)

    joint              = convert(JointNormal, μs_fw_in, Σs_fw_in)
    (μ_fw_in, Σ_fw_in) = mean_cov(joint)
    ds                 = dimensionalities(joint)
    
    μ_fw_out = A * μ_fw_in + b
    Σ_fw_out = A * Σ_fw_in * A'
    C_fw = Σ_fw_in * A'

    # RTS Smoother
    (μ_bw_out, Σ_bw_out) = mean_cov(m_out)
    (μ_in, Σ_in) = smoothRTS(μ_fw_out, Σ_fw_out, C_fw, μ_fw_in, Σ_fw_in, μ_bw_out, Σ_bw_out)

    return JointNormal(MvNormalMeanCovariance(μ_in, Σ_in), ds)
end

@marginalrule DeltaFn{f}(:ins) (m_out::NormalDistributionsFamily, m_ins::ManyOf{1, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {f, M <: Linearization} = begin
    # Approximate joint inbounds

    (μ_fw_in, Σ_fw_in) = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
    (A, b) = localLinearizationSingleIn(f, μ_fw_in)
    # (μ_fw_in, Σ_fw_in, _) = mean_cov(convert(JointNormal, μs_fw_in, Σs_fw_in))
    μ_fw_out = A * μ_fw_in + b
    Σ_fw_out = A * Σ_fw_in * A'
    C_fw = Σ_fw_in * A'

    # RTS Smoother
    (μ_bw_out, Σ_bw_out) = mean_cov(m_out)
    (μ_in, Σ_in) = smoothRTS(μ_fw_out, Σ_fw_out, C_fw, μ_fw_in, Σ_fw_in, μ_bw_out, Σ_bw_out)

    dist = convert(promote_variate_type(variate_form(μ_in), NormalMeanVariance), μ_in, Σ_in)
    ds   = [ (length(μ_in),) ]

    return JointNormal(dist, ds)
end

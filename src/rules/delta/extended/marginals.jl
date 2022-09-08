export marginalrule

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::NTuple{N, Any}, meta::DeltaExtended) where {f, N} = begin
    # Approximate joint inbounds
    (μs_fw_in, Σs_fw_in) = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
    (A, b) = localLinearizationMultiIn(f, μs_fw_in)

    (μ_fw_in, Σ_fw_in, _) = concatenateGaussianMV(μs_fw_in, Σs_fw_in)
    μ_fw_out = A * μ_fw_in + b
    Σ_fw_out = A * Σ_fw_in * A'
    C_fw = Σ_fw_in * A'

    # RTS Smoother
    (μ_bw_out, Σ_bw_out) = mean_cov(m_out)
    (μ_in, Σ_in) = smoothRTS(μ_fw_out, Σ_fw_out, C_fw, μ_fw_in, Σ_fw_in, μ_bw_out, Σ_bw_out)

    return MvNormalMeanCovariance(μ_in, Σ_in)
end

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::NTuple{1, Any}, meta::DeltaExtended) where {f} = begin
    # Approximate joint inbounds

    # (μs_fw_in, Σs_fw_in) = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
    (μs_fw_in, Σs_fw_in) = mean_cov(first(m_ins)) # Returns arrays with individual means and covariances
    (A, b) = localLinearizationSingleIn(f, μs_fw_in)
    # (μ_fw_in, Σ_fw_in, _) = concatenateGaussianMV(μs_fw_in, Σs_fw_in)
    μ_fw_in, Σ_fw_in = μs_fw_in, Σs_fw_in
    μ_fw_out = A * μ_fw_in + b
    Σ_fw_out = A * Σ_fw_in * A'
    C_fw = Σ_fw_in * A'

    # RTS Smoother
    (μ_bw_out, Σ_bw_out) = mean_cov(m_out)
    (μ_in, Σ_in) = smoothRTS(μ_fw_out, Σ_fw_out, C_fw, μ_fw_in, Σ_fw_in, μ_bw_out, Σ_bw_out)

    return MvNormalMeanCovariance(μ_in, Σ_in)
end

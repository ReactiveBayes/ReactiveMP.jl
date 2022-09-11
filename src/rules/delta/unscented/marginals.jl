export marginalrule

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::NTuple{N, Any}, meta::DeltaUnscented) where {f, N} = begin

    # Approximate joint inbounds
    (μs_fw_in, Σs_fw_in) = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
    (μ_tilde, Σ_tilde, C_tilde) =
        unscentedStatistics(μs_fw_in, Σs_fw_in, f; alpha = meta.alpha, beta = meta.beta, kappa = meta.kappa)

    # RTS smoother
    (μ_fw_in, Σ_fw_in, ds) = concatenateGaussianMV(μs_fw_in, Σs_fw_in)
    (μ_bw_out, Σ_bw_out) = mean_cov(m_out)
    (μ_in, Σ_in) = smoothRTS(μ_tilde, Σ_tilde, C_tilde, μ_fw_in, Σ_fw_in, μ_bw_out, Σ_bw_out)
    return DeltaMarginal(MvNormalMeanCovariance(μ_in, Σ_in), ds)
end

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::NTuple{1, Any}, meta::DeltaUnscented) where {f} = begin
    # Approximate joint inbounds
    μ_fw_in, Σ_fw_in = collectStatistics(m_ins...)

    (μ_tilde, Σ_tilde, C_tilde) =
        unscentedStatistics(μ_fw_in, Σ_fw_in, f; alpha = meta.alpha, beta = meta.beta, kappa = meta.kappa)
    # RTS smoother
    (μ_bw_out, Σ_bw_out) = mean_cov(m_out)

    (μ_in, Σ_in) = smoothRTS(μ_tilde, Σ_tilde, C_tilde, μ_fw_in, Σ_fw_in, μ_bw_out, Σ_bw_out)

    return DeltaMarginal(MvNormalMeanCovariance(μ_in, Σ_in), [(length(μ_in),)])
end

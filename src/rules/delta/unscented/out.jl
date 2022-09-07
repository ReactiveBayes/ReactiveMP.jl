# single input
@rule DeltaFn{f}(:out, Marginalisation) (
    m_ins::NTuple{1, NormalDistributionsFamily},
    meta::DeltaUnscented{T}
) where {f, T} = begin
    (μ_fw_in1, Σ_fw_in1) = mean_cov(first(m_ins))
    @show (μ_tilde, Σ_tilde, _) =
        unscentedStatistics(μ_fw_in1, Σ_fw_in1, f; alpha = meta.alpha, beta = meta.beta, kappa = meta.kappa)

    F = size(μ_tilde, 1) == 1 ? Univariate : Multivariate
    return convert(promote_variate_type(F, NormalMeanVariance), μ_tilde, Σ_tilde)
end

# multiple input; this should be called
@rule DeltaFn{f}(:out, Marginalisation) (
    m_ins::NTuple{N, NormalDistributionsFamily},
    meta::DeltaUnscented{T}
) where {f, N, T} =
    begin
        @show "ignored"
        (μs_in, Σs_in) = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
        (μ_tilde, Σ_tilde, _) =
            unscentedStatistics(μs_in, Σs_in, f; alpha = meta.alpha, beta = meta.beta, kappa = meta.kappa)

        F = size(μ_tilde, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), μ_tilde, Σ_tilde)
    end

# Why this method is being called?
@rule DeltaFn{f}(:out, Marginalisation) (
    m_out::NormalDistributionsFamily,
    m_ins::NTuple{N, NormalDistributionsFamily},
    meta::DeltaUnscented{T}
) where {f, N, T} =
    begin
        @show "why am i called"
        # TODO: ASK Dmitry
        (μs_in, Σs_in) = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
        (μ_tilde, Σ_tilde, _) =
            unscentedStatistics(μs_in, Σs_in, f; alpha = meta.alpha, beta = meta.beta, kappa = meta.kappa)

        F = size(μ_tilde, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), μ_tilde, Σ_tilde)
    end

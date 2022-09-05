# known inverse, single input
@rule DeltaFn{f}((:in, _), Marginalisation) (
    m_out::Any,
    m_ins::Nothing,
    meta::DeltaUnscented{T}
) where {f, T <: Function} =
    begin
        @show "in"
        (μ_bw_out, Σ_bw_out) = mean_cov(m_out)
        (μ_tilde, Σ_tilde, _) = unscentedStatistics(
            μ_bw_out,
            Σ_bw_out,
            meta.inverse;
            alpha = meta.alpha,
            beta = meta.beta,
            kappa = meta.kappa
        )

        F = size(μ_tilde, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), μ_tilde, Σ_tilde)
    end

# I expect this rule to be called when inverse is given, known inverse
@rule DeltaFn{f}((:in, k), Marginalisation) (m_out::Any, m_ins::Any, meta::DeltaUnscented{T}) where {f, T} =
    begin
        @show "multiple input backward"
        (ms, Vs) = collectStatistics(m_in, q_ins...) # Returns arrays with individual means and covariances
        (A, b) = localLinearization(meta.inverse, ms)
        (mc, Vc) = concatenateGaussianMV(ms, Vs)
        m = A * m_out + b
        V = A * V_out * A'

        F = size(m, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

# why this method is called?, known inverse
@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::DeltaUnscented{T}) where {f, T} =
    begin
        @show meta.inverse[k](mean(q_ins))

        @show (ms, Vs) = mean_cov(q_ins)
        (m_tilde, V_tilde, _) =
            unscentedStatistics(ms, Vs, meta.inverse[k]; alpha = meta.alpha, beta = meta.beta, kappa = meta.kappa)

        F = size(m_tilde, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), m_tilde, V_tilde)
    end

@rule DeltaFn{f}((:in, _), Marginalisation) (
    m_out::Any,
    m_in::Nothing,
    meta::DeltaUnscented{T}
) where {f, T <: Nothing} =
    begin
        @show "called"

        F = size(m, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

# why this method is called for single input unknown inverse
# TODO: This won't work, to discuss
@rule DeltaFn{f}((:in, k), Marginalisation) (
    q_ins::Any,
    m_in::NormalDistributionsFamily,
    meta::DeltaUnscented{T}
) where {f, T <: Nothing} =
    begin
        (μ_fw_in1, Σ_fw_in1) = mean_cov(q_ins)
        @show (μ_tilde, Σ_tilde, C_tilde) =
            unscentedStatistics(μ_fw_in1, Σ_fw_in1, f; alpha = meta.alpha, beta = meta.beta, kappa = meta.kappa)

        # RTS smoother
        (μ_bw_out, Σ_bw_out) = mean_cov(q_ins)
        (μ_bw_in1, Σ_bw_in1) = smoothRTSMessage(μ_tilde, Σ_tilde, C_tilde, μ_fw_in1, Σ_fw_in1, μ_bw_out, Σ_bw_out)

        F = size(μ_bw_in1, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalWeightedMeanPrecision), μ_bw_in1, Σ_bw_in1)
    end

# why this method is called?, unknown inverse
@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::DeltaUnscented{T}) where {f, T <: Nothing} =
    begin
        @show "called"

        F = size(ξ, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalWeightedMeanPrecision), ξ, Λ)
    end

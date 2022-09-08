# known inverse, single input
@rule DeltaFn{f}((:in, _), Marginalisation) (
    m_out::Any,
    m_ins::Nothing,
    meta::DeltaUnscented{T}
) where {f, T <: Function} =
    begin
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

# I expect this rule to be called when inverse is given
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

@rule DeltaFn{f}((:in, k), Marginalisation) (m_out::Any, m_ins::NTuple{N, Any}, meta::DeltaUnscented{T}) where {f, N, T <: Any} =
begin

    (μs_in, Σs_in) = collectStatistics(m_out, m_ins...)
    (m_tilde, V_tilde, _) = unscentedStatistics(μs_in, Σs_in, meta.inverse[k]; alpha=meta.alpha, beta=meta.beta, kappa=meta.kappa)

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

# TODO: This won't work for single input, to discuss
@rule DeltaFn{f}((:in, k), Marginalisation) (
    q_ins::Any,
    m_in::NormalDistributionsFamily,
    meta::DeltaUnscented{T}
) where {f, T <: Nothing} =
    begin
        inx = k
        @show μ_in, Σ_in = mean_cov(q_ins)
        # ds = [(ndims(m_in),) for _ in 1:Int(round(length(μ_in) / ndims(m_in)))] # sorry, I assumed that all dimensions on the interfaces are same
        ds = [(length(mean(m_in)),) for _ in 1:Int(round(length(μ_in) / length(mean(m_in))))] # sorry, I assumed that all dimensions on the interfaces are same

        
        # Marginalize joint belief on in's
        (μ_inx, Σ_inx) = marginalizeGaussianMV(μ_in, Σ_in, ds, inx) # Marginalization is overloaded on VariateType V
        Λ_inx = cholinv(Σ_inx) # Convert to canonical statistics
        ξ_inx = Λ_inx*μ_inx

        # TODO: ugly
        ξ_inx = size(ξ_inx, 1) == 1 ? first(ξ_inx) : ξ_inx
        Λ_inx = size(Λ_inx, 1) == 1 ? first(Λ_inx) : Λ_inx

        # Divide marginal on inx by forward message
        (ξ_fw_inx, Λ_fw_inx) = weightedmean_precision(m_in)
        ξ_bw_inx = ξ_inx - ξ_fw_inx
        Λ_bw_inx = Λ_inx - Λ_fw_inx # Note: subtraction might lead to posdef violations

        F = size(ξ_bw_inx, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalWeightedMeanPrecision), ξ_bw_inx, Λ_bw_inx)
    end

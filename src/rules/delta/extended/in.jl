@rule DeltaFn{f}((:in, _), Marginalisation) (
    m_out::Any,
    m_ins::Nothing,
    meta::DeltaExtended{T}
) where {f, T <: Function} =
    begin
        μ_out, Σ_out = mean_cov(m_out)
        (A, b) = localLinearizationSingleIn(meta.inverse, μ_out)
        m = A * μ_out + b
        V = A * Σ_out * A'

        F = size(m, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

@rule DeltaFn{f}((:in, k), Marginalisation) (
    m_out::Any,
    m_ins::ManyOf{N, Any},
    meta::DeltaExtended{T}
) where {f, N, T <: Any} =
    begin
        (μs_in, Σs_in) = collectStatistics(m_out, m_ins...)
        (A, b) = localLinearizationMultiIn(meta.inverse[k], μs_in)
        (μ_in, Σ_in, _) = concatenateGaussianMV(μs_in, Σs_in)
        m = A * μ_in + b
        V = A * Σ_in * A'

        F = size(m, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::DeltaExtended{T}) where {f, T <: Nothing} =
    begin
        # we need dimension of the interface variables
        # Marginalize joint belief on in's
        inx = k
        q_ins, ds = q_ins.dist, q_ins.ds
        μ_in, Σ_in = mean_cov(q_ins)

        (μ_inx, Σ_inx) = marginalizeGaussianMV(μ_in, Σ_in, ds, inx)
        Λ_inx = cholinv(Σ_inx) # Convert to canonical statistics
        ξ_inx = Λ_inx * μ_inx

        F = variate_form(m_in)

        ξ_inx, Λ_inx = F == Univariate ? (first(ξ_inx), first(Λ_inx)) : (ξ_inx, Λ_inx)

        # Divide marginal on inx by forward message
        (ξ_fw_inx, Λ_fw_inx) = weightedmean_precision(m_in)

        ξ_bw_inx = ξ_inx - ξ_fw_inx
        Λ_bw_inx = Λ_inx - Λ_fw_inx # Note: subtraction might lead to posdef violations

        return convert(promote_variate_type(F, NormalWeightedMeanPrecision), ξ_bw_inx, Λ_bw_inx)
    end

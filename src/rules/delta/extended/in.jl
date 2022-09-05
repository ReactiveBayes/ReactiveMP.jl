# known inverse, single input
@rule DeltaFn{f}((:in, _), Marginalisation) (
    m_out::Any,
    m_ins::Nothing,
    meta::DeltaExtended{T}
) where {f, T <: Function} =
    begin
        @show "in"
        μ_out, Σ_out = mean_cov(m_out)
        (A, b) = localLinearizationSingleIn(meta.inverse, μ_out)
        m = A * μ_out + b
        V = A * Σ_out * A'

        F = size(m, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

# I expect this rule to be called when inverse is given, known inverse
@rule DeltaFn{f}((:in, k), Marginalisation) (m_out::Any, m_ins::Any, meta::DeltaExtended{T}) where {f, T} =
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
@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::DeltaExtended{T}) where {f, T} =
    begin
        @show meta.inverse[k](mean(q_ins))
        (A, b) = localLinearizationMultiIn(meta.inverse[k], mean(q_ins))
        (mc, Vc) = concatenateGaussianMV(ms, Vs)
        m = A * m_out + b
        V = A * V_out * A'

        F = size(m, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

@rule DeltaFn{f}((:in, _), Marginalisation) (
    m_out::Any,
    m_in::Nothing,
    meta::DeltaExtended{T}
) where {f, T <: Nothing} =
    begin
        μ_out, Σ_out = mean_cov(m_out)
        (A, b) = localLinearizationSingleIn(meta.inverse, μ_out)
        m = A * μ_out + b
        V = A * Σ_out * A'

        F = size(m, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

# TODO: ugly af
# why this method is called?, unknown inverse
@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::DeltaExtended{T}) where {f, T <: Nothing} =
    begin

        # we need dimension of the interface variables
        # Marginalize joint belief on in's
        inx = k
        μ_in, Σ_in = mean_cov(q_ins)
        ds = [(ndims(m_in),) for _ in 1:Int(round(length(μ_in) / ndims(m_in)))] # sorry, I assumed that all dimensions on the interfaces are same
        (μ_inx, Σ_inx) = marginalizeGaussianMV(μ_in, Σ_in, ds, inx)
        Λ_inx = cholinv(Σ_inx) # Convert to canonical statistics
        ξ_inx = Λ_inx * μ_inx

        # Divide marginal on inx by forward message
        (ξ_fw_inx, Λ_fw_inx) = weightedmean_precision(m_in)
        ξ_bw_inx = ξ_inx - ξ_fw_inx
        Λ_bw_inx = Λ_inx - Λ_fw_inx # Note: subtraction might lead to posdef violations

        @show ξ_bw_inx, Λ_bw_inx
        F = size(ξ_bw_inx, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalWeightedMeanPrecision), ξ_bw_inx, Λ_bw_inx)
    end

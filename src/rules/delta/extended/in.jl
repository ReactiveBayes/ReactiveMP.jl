# known inverse, single input
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
    
# why this method is called?, known inverse should be just m_out::Any, m_ins
@rule DeltaFn{f}((:in, k), Marginalisation) (m_out::Any, m_ins::Any, meta::DeltaExtended{T}) where {f, T <: Any} =
    begin

        @show q_ins
        (A, b) = localLinearizationMultiIn(meta.inverse[k], mean(q_ins))
        (mc, Vc) = concatenateGaussianMV(ms, Vs)
        m = A * m_out + b
        V = A * V_out * A'

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

        ds = [(length(mean(m_in)),) for _ in 1:Int(round(length(μ_in) / length(mean(m_in))))] # sorry, I assumed that all dimensions on the interfaces are same
        (μ_inx, Σ_inx) = marginalizeGaussianMV(μ_in, Σ_in, ds, inx)
        Λ_inx = cholinv(Σ_inx) # Convert to canonical statistics
        ξ_inx = Λ_inx * μ_inx

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

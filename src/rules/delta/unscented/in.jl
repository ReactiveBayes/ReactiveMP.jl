
# unknown inverse
@rule DeltaFn{f}((:in, k), Marginalisation) (
    q_ins::JointNormal,
    m_in::NormalDistributionsFamily,
    meta::DeltaMeta{M, I}
) where {f, M <: Unscented, I <: Nothing} = begin
    # Divide marginal on inx by forward message
    ξ_inx, Λ_inx       = weightedmean_precision(getmarginal(q_ins, k))
    ξ_fw_inx, Λ_fw_inx = weightedmean_precision(m_in)

    ξ_bw_inx = ξ_inx - ξ_fw_inx
    Λ_bw_inx = Λ_inx - Λ_fw_inx # Note: subtraction might lead to posdef violations

    return convert(promote_variate_type(variate_form(ξ_inx), NormalWeightedMeanPrecision), ξ_bw_inx, Λ_bw_inx)
end

# known inverse, single input
@rule DeltaFn{f}((:in, _), Marginalisation) (m_out::NormalDistributionsFamily, m_ins::Nothing, meta::DeltaMeta{M, I}) where {f, M <: Unscented, I <: Function} =
    begin
        (μ_bw_out, Σ_bw_out)  = mean_cov(m_out)
        (μ_tilde, Σ_tilde, _) = unscented_statistics(getmethod(meta), μ_bw_out, Σ_bw_out, getinverse(meta))
        return convert(promote_variate_type(variate_form(μ_tilde), NormalMeanVariance), μ_tilde, Σ_tilde)
    end

@rule DeltaFn{f}((:in, k), Marginalisation) (
    m_out::NormalDistributionsFamily,
    m_ins::ManyOf{N, NormalDistributionsFamily},
    meta::DeltaMeta{M, I}
) where {f, N, M <: Unscented, L, I <: NTuple{L, Function}} = begin
    (μs_in, Σs_in)        = collectStatistics(m_out, m_ins...)
    (μ_tilde, Σ_tilde, _) = unscented_statistics(getmethod(meta), μs_in, Σs_in, getinverse(meta, k))
    return convert(promote_variate_type(variate_form(μ_tilde), NormalMeanVariance), μ_tilde, Σ_tilde)
end


@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::JointNormal, m_in::NormalDistributionsFamily, meta::DeltaMeta{M, Nothing}) where {f, M <: Linearization} =
    begin
        # Divide marginal on inx by forward message
        ξ_inx, Λ_inx       = weightedmean_precision(getmarginal(q_ins, k))
        ξ_fw_inx, Λ_fw_inx = weightedmean_precision(m_in)

        ξ_bw_inx = ξ_inx - ξ_fw_inx
        Λ_bw_inx = Λ_inx - Λ_fw_inx # Note: subtraction might lead to posdef violations

        return convert(promote_variate_type(variate_form(ξ_inx), NormalWeightedMeanPrecision), ξ_bw_inx, Λ_bw_inx)
    end

@rule DeltaFn{f}((:in, _), Marginalisation) (
    m_out::NormalDistributionsFamily,
    m_ins::Nothing,
    meta::DeltaMeta{M, I}
) where {f, M <: Linearization, I <: Function} = begin
    return approximate(getmethod(meta), getinverse(meta), (m_out,))
end

@rule DeltaFn{f}((:in, k), Marginalisation) (
    m_out::NormalDistributionsFamily,
    m_ins::ManyOf{N, NormalDistributionsFamily},
    meta::DeltaMeta{M, I}
) where {f, N, M <: Linearization, L, I <: NTuple{L, Function}} = begin
    return approximate(getmethod(meta), getinverse(meta, k), (m_out, m_ins...))
end

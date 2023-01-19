
# unknown inverse
@rule DeltaFn((:in, k), Marginalisation) (q_ins::JointNormal, m_in::NormalDistributionsFamily, meta::DeltaMeta{M, I}) where {M <: Unscented, I <: Nothing} = begin
    # Divide marginal on inx by forward message
    ξ_inx, Λ_inx       = weightedmean_precision(getmarginal(q_ins, k))
    ξ_fw_inx, Λ_fw_inx = weightedmean_precision(m_in)

    ξ_bw_inx = ξ_inx - ξ_fw_inx
    Λ_bw_inx = Λ_inx - Λ_fw_inx # Note: subtraction might lead to posdef violations

    return convert(promote_variate_type(variate_form(ξ_inx), NormalWeightedMeanPrecision), ξ_bw_inx, Λ_bw_inx)
end

# known inverse, single input
@rule DeltaFn((:in, _), Marginalisation) (m_out::NormalDistributionsFamily, m_ins::Nothing, meta::DeltaMeta{M, I}) where {M <: Unscented, I <: Function} = begin
    return approximate(getmethod(meta), getnodefn(Val(:in)), (m_out,))
end

@rule DeltaFn((:in, k), Marginalisation) (
    m_out::NormalDistributionsFamily, m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M, I}
) where {N, M <: Unscented, L, I <: NTuple{L, Function}} = begin
    return approximate(getmethod(meta), getnodefn(Val(:in), k), (m_out, m_ins...))
end

# #Test with gp 
@rule DeltaFn((:in, _), Marginalisation) (m_out::ContinuousUnivariateLogPdf, m_ins::Nothing, meta::Tuple{ProcessMeta, DeltaMeta{M,I}}) where {M <: Unscented, I <: Function}= begin 
    μ_out = mean(m_out)
    var_out = var(m_out)
    # exp_mean = exp(μ_out)
    # exp_var = exp(μ_out + var_out)
    # approx_out = NormalMeanVariance(exp_mean,exp_var)
    approx_out = NormalMeanVariance(μ_out,var_out)
    return approximate(getmethod(meta[2]), getnodefn(Val(:in)), (approx_out,))
end


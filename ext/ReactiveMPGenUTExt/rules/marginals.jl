@marginalrule DeltaFn(:ins) (m_out::NormalDistributionsFamily, m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {N, M <: GenUnscented} = begin
    # Approximate joint inbounds
    statistics = mean_cov.(m_ins)
    μs_fw_in = first.(statistics)
    Σs_fw_in = last.(statistics)
	S_diag = map(x -> zero(x), μs_fw_in)
    K_diag = map(v -> 3 .*GeneralizedUnscented._diag(v).^2, Σs_fw_in)
	
    sizes = size.(m_ins)
    (stats, C_tilde) = GeneralizedUnscented.transform_statistics_with_cross_covariance(GeneralizedUnscented.StatisticsOrder(2), ReactiveMP.getmethod(meta), getnodefn(meta, Val(:out)), μs_fw_in, Σs_fw_in, S_diag, K_diag)
    μ_tilde, Σ_tilde = stats

	joint              = convert(JointNormal, μs_fw_in, Σs_fw_in)
    (μ_fw_in, Σ_fw_in) = mean_cov(joint)
    ds                 = ExponentialFamily.dimensionalities(joint)

    # Apply the RTS smoother
    (μ_bw_out, Σ_bw_out) = mean_cov(m_out)
    (μ_in, Σ_in)         = ReactiveMP.smoothRTS(μ_tilde, Σ_tilde, C_tilde', μ_fw_in, Σ_fw_in, μ_bw_out, Σ_bw_out)

    dist = convert(promote_variate_type(typeof(μ_in), NormalMeanVariance), μ_in, Σ_in)
	# dist = convert(promote_variate_type(Float64, NormalMeanPrecision), μ_in, Σ_in)

    return JointNormal(dist, sizes)
end
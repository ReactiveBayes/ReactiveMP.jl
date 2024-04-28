
# most of routines are ported from ForneyLab.jl

@marginalrule DeltaFn(:ins) (m_out::NormalDistributionsFamily, m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {N, M <: Unscented} = begin
    # Approximate joint inbounds
    statistics = mean_cov.(m_ins)
    μs_fw_in = first.(statistics)
    Σs_fw_in = last.(statistics)
    sizes = size.(m_ins)

    (μ_tilde, Σ_tilde, C_tilde) = unscented_statistics(getmethod(meta), getnodefn(meta, Val(:out)), μs_fw_in, Σs_fw_in)

    joint              = convert(JointNormal, μs_fw_in, Σs_fw_in)
    (μ_fw_in, Σ_fw_in) = mean_cov(joint)
    ds                 = ExponentialFamily.dimensionalities(joint)

    # Apply the RTS smoother
    (μ_bw_out, Σ_bw_out) = mean_cov(m_out)
    (μ_in, Σ_in)         = smoothRTS(μ_tilde, Σ_tilde, C_tilde, μ_fw_in, Σ_fw_in, μ_bw_out, Σ_bw_out)

    dist = convert(promote_variate_type(typeof(μ_in), NormalMeanVariance), μ_in, Σ_in)

    return JointNormal(dist, sizes)
end

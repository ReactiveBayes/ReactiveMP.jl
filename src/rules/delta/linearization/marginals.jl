# most of the routines are ported directly from ForneyLab.jl
@marginalrule DeltaFn(:ins) (m_out::NormalDistributionsFamily, m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where { N, M <: Linearization } = begin
    # Approximate joint inbounds
    # Collect individual means and covariances
    statistics = mean_cov.(m_ins)
    μs_fw_in = first.(statistics)
    Σs_fw_in = last.(statistics)
    sizes = size.(m_ins)

    # Calculate local linear components
    (A, b) = approximate(getmethod(meta), getnodefn(Val(:out)), μs_fw_in)

    # Invoke the "concatenated" messages in the local linearization
    joint              = convert(JointNormal, μs_fw_in, Σs_fw_in)
    (μ_fw_in, Σ_fw_in) = mean_cov(joint)

    μ_fw_out = A * μ_fw_in + b
    Σ_fw_out = A * Σ_fw_in * A'
    C_fw     = Σ_fw_in * A'

    # Apply the RTS Smoother
    (μ_bw_out, Σ_bw_out) = mean_cov(m_out)
    (μ_in, Σ_in) = smoothRTS(μ_fw_out, Σ_fw_out, C_fw, μ_fw_in, Σ_fw_in, μ_bw_out, Σ_bw_out)

    dist = convert(promote_variate_type(variate_form(μ_in), NormalMeanVariance), μ_in, Σ_in)

    return convert(JointNormal, dist, sizes)
end

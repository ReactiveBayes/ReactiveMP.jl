
# Belief Propagation                #
# --------------------------------- #
@rule NormalMeanVariance(:v, Marginalisation) (m_out::PointMass, m_μ::UnivariateNormalDistributionsFamily) = begin
    m_out_mean        = mean(m_out)
    m_μ_mean, m_μ_cov = mean_cov(m_μ)

    return ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(m_μ_cov + x) / 2 - 1 / (2 * (x + m_μ_cov)) * (m_out_mean - m_μ_mean)^2)
end

@rule NormalMeanVariance(:v, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_μ::PointMass) = begin
    m_out_mean, m_out_cov = mean_cov(m_out)
    m_μ_mean              = mean(m_μ)

    return ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(m_out_cov + x) / 2 - 1 / (2 * (x + m_out_cov)) * (m_out_mean - m_μ_mean)^2)
end

@rule NormalMeanVariance(:v, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_μ::UnivariateNormalDistributionsFamily) = begin
    m_out_mean, m_out_cov = mean_cov(m_out)
    m_μ_mean, m_μ_cov     = mean_cov(m_μ)

    return ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(m_out_cov + m_μ_cov + x) / 2 - 1 / (2 * (m_out_cov + m_μ_cov + x)) * (m_out_mean - m_μ_mean)^2)
end

# Variational                       #
# --------------------------------- #
@rule NormalMeanVariance(:v, Marginalisation) (q_out::Any, q_μ::Any) = begin
    mμ, vμ = mean_var(q_μ)
    mx, vx = mean_var(q_out)

    θ = (mx^2 + vx - 2 * mx * mμ + mμ^2 + vμ) / 2
    α = convert(typeof(θ), -0.5)

    return GammaInverse(α, θ; check_args = false)
end

# TODO: substitute
@rule NormalMeanVariance(:v, Marginalisation) (q_out_μ::MultivariateNormalDistributionsFamily,) = begin
    m, V = mean_cov(q_out_μ)
    mx = m[1]
    mm = m[2]
    vxx = V[1, 1]
    vmx = V[1, 2]
    vmm = V[2, 2]

    θ = ((mx - mm)^2 + vxx - 2 * vmx + vmm) / 2
    α = convert(typeof(θ), -0.5)

    return GammaInverse(α, θ; check_args = false)
end

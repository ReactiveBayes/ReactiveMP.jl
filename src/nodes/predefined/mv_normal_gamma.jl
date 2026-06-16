export MvNormalGamma

import SpecialFunctions: loggamma, digamma
import StatsFuns: log2π

# `MvNormalGamma` is defined as a distribution in ExponentialFamily; here we register it as a
# (prior) factor node so it can be used in models, e.g. `w ~ MvNormalGamma(μ, Λ, α, β)`.
@node MvNormalGamma Stochastic [out, μ, Λ, α, β]

# Average energy U = −E_q(out)[log MvNormalGamma(out; μ₀, Λ₀, α₀, β₀)]: the cross-entropy of the
# posterior marginal against the prior factor, needed for the Bethe free energy. Uses the joint
# moments E[log γ] = ψ(α)−log β, E[γ] = α/β, and
#   E[γ (θ−μ₀)ᵀΛ₀(θ−μ₀)] = tr(Λ₀ Λ⁻¹) + (α/β)(μ−μ₀)ᵀΛ₀(μ−μ₀).
@average_energy MvNormalGamma (
    q_out::MvNormalGamma,
    q_μ::PointMass,
    q_Λ::PointMass,
    q_α::PointMass,
    q_β::PointMass,
) = begin
    μ_q, Λ_q, α_q, β_q = params(q_out)
    μ0, Λ0, α0, β0 = mean(q_μ), mean(q_Λ), mean(q_α), mean(q_β)
    d = length(μ0)

    Δ      = μ_q - μ0
    E_logγ = digamma(α_q) - log(β_q)
    E_γ    = α_q / β_q
    E_quad  = tr(Λ0 * cholinv(Λ_q)) + E_γ * dot(Δ, Λ0, Δ)

    E_logp =
        α0 * log(β0) + logdet(Λ0) / 2 - loggamma(α0) - (d / 2) * log2π +
        (α0 + d / 2 - 1) * E_logγ - β0 * E_γ - E_quad / 2

    return -E_logp
end

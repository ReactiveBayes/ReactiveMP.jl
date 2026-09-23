@define_factor_node(node = MvNormalGamma, type = Stochastic, interfaces = [:out, :μ, :Λ, :α, :β])

# -E[log MvNormalGamma(out | μ₀, Λ₀, α₀, β₀)] for known parameters, with E[log γ] = ψ(α) - log β,
# E[γ] = α/β and E[γ (θ - μ₀)ᵀΛ₀(θ - μ₀)] = tr(Λ₀ Λ⁻¹) + E[γ] (μ - μ₀)ᵀΛ₀(μ - μ₀) under q(out).
function mv_normal_gamma_energy(q_out, μ0, Λ0, α0, β0)
    μ, Λ, α, β = params(q_out)
    d = length(μ0)
    Δ = μ - μ0
    E_logγ, E_γ = digamma(α) - log(β), α / β
    E_quad = tr(Λ0 * cholinv(Λ)) + E_γ * dot(Δ, Λ0, Δ)
    # (2α₀ + d - 2)/2 is α₀ + d/2 - 1 in α₀'s float type.
    rest = -2 * (α0 * log(β0) + logdet(Λ0) / 2 - loggamma(α0) + (2α0 + d - 2) / 2 * E_logγ - β0 * E_γ - E_quad / 2)
    return gaussian_energy(d, rest)
end

@define_average_energy(
    node = MvNormalGamma,
    args = (q[:out]::MvNormalGamma, q[:μ]::PointMass, q[:Λ]::PointMass, q[:α]::PointMass, q[:β]::PointMass),
    body = (args) -> mv_normal_gamma_energy(args.q[:out], mean(args.q[:μ]), mean(args.q[:Λ]), mean(args.q[:α]), mean(args.q[:β])),
)

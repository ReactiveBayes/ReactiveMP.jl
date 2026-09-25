@define_factor_node(
    node = MvNormalWeightedMeanPrecision, type = Stochastic,
    interfaces = [:out, (:ξ, aliases = [:xi, :weightedmean]), (:Λ, aliases = [:invcov, :precision])],
)

# -log N(x; Λ⁻¹ξ, Λ⁻¹) = (d log 2π - log |Λ| + xᵀΛx - 2xᵀξ + ξᵀΛ⁻¹ξ) / 2, in expectation over
# independent marginals. E[log |Λ|] is `mean(logdet, q_Λ)`, defined for a point mass and for a
# Wishart.
@define_average_energy(
    node = MvNormalWeightedMeanPrecision,
    args = (q[:out]::Any, q[:ξ]::Any, q[:Λ]::Any),
    body = (args) -> begin
        m_ξ, V_ξ = mean_cov(args.q[:ξ])
        m_out, V_out = mean_cov(args.q[:out])
        gaussian_energy(
            ndims(args.q[:out]),
            tr(mean(args.q[:Λ]) * (m_out * m_out' + V_out)) - 2 * dot(m_out, m_ξ) + tr(mean(cholinv, args.q[:Λ]) * (m_ξ * m_ξ' + V_ξ)) - mean(logdet, args.q[:Λ]),
        )
    end,
)

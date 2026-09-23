@define_factor_node(node = MvNormalMeanCovariance, type = Stochastic, interfaces = [:out, (:μ, aliases = [:mean]), (:Σ, aliases = [:cov])])

# (d log 2π + E[log |Σ|] + tr(E[Σ⁻¹] E[(out - μ)(out - μ)ᵀ])) / 2
mv_normal_mean_covariance_energy(d, q_Σ, S) = (d * log2π + mean(logdet, q_Σ) + tr(mean(cholinv, q_Σ) * S)) / 2

@define_average_energy(
    node = MvNormalMeanCovariance,
    args = (q[:out]::Any, q[:μ]::Any, q[:Σ]::Any),
    body = (args) -> mv_normal_mean_covariance_energy(ndims(args.q[:out]), args.q[:Σ], difference_moment(args.q[:out], args.q[:μ])),
)

@define_average_energy(
    node = MvNormalMeanCovariance,
    args = (q[:out, :μ]::Any, q[:Σ]::Any),
    body = (args) -> mv_normal_mean_covariance_energy(div(ndims(args.q[:out, :μ]), 2), args.q[:Σ], difference_moment(args.q[:out, :μ])),
)

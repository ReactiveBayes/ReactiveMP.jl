@define_factor_node(
    node = MvNormalMeanScaleMatrixPrecision, type = Stochastic,
    interfaces = [:out, (:μ, aliases = [:mean]), (:γ, aliases = [:scale]), (:G, aliases = [:matrix])],
)

# The multivariate normal with precision γG:
# (d log 2π - d E[log γ] - E[log |G|] + E[γ] tr(E[G] S)) / 2.
mv_normal_mean_scale_matrix_precision_energy(d, q_γ, q_G, S) =
    (d * log2π - d * mean(log, q_γ) - mean(logdet, q_G) + mean(q_γ) * tr(mean(q_G) * S)) / 2

@define_average_energy(
    node = MvNormalMeanScaleMatrixPrecision,
    args = (q[:out]::Any, q[:μ]::Any, q[:γ]::Any, q[:G]::Any),
    body = (args) -> mv_normal_mean_scale_matrix_precision_energy(
        ndims(args.q[:out]), args.q[:γ], args.q[:G], difference_moment(args.q[:out], args.q[:μ]),
    ),
)

@define_average_energy(
    node = MvNormalMeanScaleMatrixPrecision,
    args = (q[:out, :μ]::Any, q[:γ]::Any, q[:G]::Any),
    body = (args) -> mv_normal_mean_scale_matrix_precision_energy(
        div(ndims(args.q[:out, :μ]), 2), args.q[:γ], args.q[:G], difference_moment(args.q[:out, :μ]),
    ),
)

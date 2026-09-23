@define_factor_node(node = MvNormalMeanScalePrecision, type = Stochastic, interfaces = [:out, (:μ, aliases = [:mean]), (:γ, aliases = [:precision])])

# The multivariate normal with precision γI: (d log 2π - d E[log γ] + E[γ] tr S) / 2.
mv_normal_mean_scale_precision_energy(d, q_γ, S) = gaussian_energy(d, mean(q_γ) * tr(S) - d * mean(log, q_γ))

@define_average_energy(
    node = MvNormalMeanScalePrecision,
    args = (q[:out]::Any, q[:μ]::Any, q[:γ]::Any),
    body = (args) -> mv_normal_mean_scale_precision_energy(ndims(args.q[:out]), args.q[:γ], difference_moment(args.q[:out], args.q[:μ])),
)

@define_average_energy(
    node = MvNormalMeanScalePrecision,
    args = (q[:out, :μ]::Any, q[:γ]::Any),
    body = (args) -> mv_normal_mean_scale_precision_energy(div(ndims(args.q[:out, :μ]), 2), args.q[:γ], difference_moment(args.q[:out, :μ])),
)

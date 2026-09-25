@define_factor_node(node = MvNormalMeanPrecision, type = Stochastic, interfaces = [:out, (:μ, aliases = [:mean]), (:Λ, aliases = [:invcov, :precision])])

# (d log 2π - E[log |Λ|] + tr(E[Λ] S)) / 2, the average energy of a multivariate normal with
# precision `Λ` and `S = E[(out - μ)(out - μ)ᵀ]`. A Wishart `q_Λ` gives E[Λ] as `df · S_Λ`
# without building the mean matrix.
mv_normal_mean_precision_energy(d, q_Λ, S) = gaussian_energy(d, tr(mean(q_Λ) * S) - mean(logdet, q_Λ))

function mv_normal_mean_precision_energy(d, q_Λ::Wishart, S)
    df, S_Λ = params(q_Λ)
    return gaussian_energy(d, df * tr(S_Λ * S) - mean(logdet, q_Λ))
end

@define_average_energy(
    node = MvNormalMeanPrecision,
    args = (q[:out]::Any, q[:μ]::Any, q[:Λ]::Any),
    body = (args) -> mv_normal_mean_precision_energy(ndims(args.q[:out]), args.q[:Λ], difference_moment(args.q[:out], args.q[:μ])),
)

@define_average_energy(
    node = MvNormalMeanPrecision,
    args = (q[:out, :μ]::Any, q[:Λ]::Any),
    body = (args) -> mv_normal_mean_precision_energy(div(ndims(args.q[:out, :μ]), 2), args.q[:Λ], difference_moment(args.q[:out, :μ])),
)

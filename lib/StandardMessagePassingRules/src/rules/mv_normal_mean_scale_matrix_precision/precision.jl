# A Gamma likelihood of γ: shape d/2 + 1, rate tr(E[G] E[(out - μ)(out - μ)ᵀ]) / 2.
scale_matrix_precision_likelihood(d, G, S) = (β = tr(G * S) / 2; GammaShapeRate(convert(typeof(β), d / 2 + 1), β))

@define_message_update_rule(
    node = MvNormalMeanScaleMatrixPrecision, target = :γ,
    args = (q[:out]::Any, q[:μ]::Any, q[:G]::Any),
    body = (args) -> scale_matrix_precision_likelihood(ndims(args.q[:μ]), mean(args.q[:G]), difference_moment(args.q[:out], args.q[:μ])),
)

@define_message_update_rule(
    node = MvNormalMeanScaleMatrixPrecision, target = :γ,
    args = (q[:out, :μ]::Any, q[:G]::Any),
    body = (args) -> scale_matrix_precision_likelihood(div(ndims(args.q[:out, :μ]), 2), mean(args.q[:G]), difference_moment(args.q[:out, :μ])),
)

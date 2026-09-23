# A Gamma likelihood of γ: shape d/2 + 1, rate tr(E[(out - μ)(out - μ)ᵀ]) / 2.
scale_precision_likelihood(d, S) = (β = tr(S) / 2; GammaShapeRate(convert(typeof(β), d / 2 + 1), β))

@define_message_update_rule(
    node = MvNormalMeanScalePrecision, target = :γ,
    args = (q[:out]::Any, q[:μ]::Any),
    body = (args) -> scale_precision_likelihood(ndims(args.q[:μ]), difference_moment(args.q[:out], args.q[:μ])),
)

@define_message_update_rule(
    node = MvNormalMeanScalePrecision, target = :γ,
    args = (q[:out, :μ]::Any,),
    body = (args) -> scale_precision_likelihood(div(ndims(args.q[:out, :μ]), 2), difference_moment(args.q[:out, :μ])),
)

# A Wishart likelihood of Λ with d + 2 degrees of freedom. Its scale matrix goes through
# `ctx.matrix_correction` first, which `nothing` leaves as it is.

@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :Λ, ctx = (:matrix_correction,),
    args = (q[:out]::Any, q[:μ]::Any),
    body = (ctx, args) -> WishartFast(ndims(args.q[:μ]) + 2, correction!(ctx.matrix_correction, difference_moment(args.q[:out], args.q[:μ]))),
)

@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :Λ, ctx = (:matrix_correction,),
    args = (q[:out, :μ]::Any,),
    body = (ctx, args) -> WishartFast(div(ndims(args.q[:out, :μ]), 2) + 2, correction!(ctx.matrix_correction, difference_moment(args.q[:out, :μ]))),
)

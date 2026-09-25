# An inverse Wishart, in `InverseWishartFast`'s parametrisation, with degrees of freedom -d,
# the likelihood of Σ.

@define_message_update_rule(
    node = MvNormalMeanCovariance, target = :Σ,
    args = (q[:out]::Any, q[:μ]::Any),
    body = (args) -> InverseWishartFast(-ndims(args.q[:μ]), difference_moment(args.q[:out], args.q[:μ])),
)

@define_message_update_rule(
    node = MvNormalMeanCovariance, target = :Σ,
    args = (q[:out, :μ]::Any,),
    body = (args) -> InverseWishartFast(-div(ndims(args.q[:out, :μ]), 2), difference_moment(args.q[:out, :μ])),
)

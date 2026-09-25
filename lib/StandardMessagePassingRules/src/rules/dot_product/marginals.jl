# The joint of the inputs with one known: the other's marginal is its message times its prior.
# `out` is univariate, as the message rule requires.

@define_marginal_update_rule(
    node = dot, target = (:in1, :in2), ctx = (:matrix_correction,),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:in1]::PointMass, m[:in2]::NormalDistributionsFamily),
    body = (ctx, args) -> promoted_cluster(
        FactorizedCluster((:in1,) => args.m[:in1], (:in2,) => prod(ClosedProd(), args.m[:in2], dot_backward(ctx, args.m[:in1], args.m[:out]))),
        args.m[:out], args.m[:in1], args.m[:in2],
    ),
)

@define_marginal_update_rule(
    node = dot, target = (:in1, :in2), ctx = (:matrix_correction,),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:in1]::NormalDistributionsFamily, m[:in2]::PointMass),
    body = (ctx, args) -> promoted_cluster(
        FactorizedCluster((:in1,) => prod(ClosedProd(), args.m[:in1], dot_backward(ctx, args.m[:in2], args.m[:out])), (:in2,) => args.m[:in2]),
        args.m[:out], args.m[:in1], args.m[:in2],
    ),
)

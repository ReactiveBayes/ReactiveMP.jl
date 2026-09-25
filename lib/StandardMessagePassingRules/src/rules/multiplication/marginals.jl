# The joint of the factors with one known: the other's marginal is its message times its prior.

@define_marginal_update_rule(
    node = *, target = (:A, :in), ctx = (:matrix_correction,),
    args = (m[:out]::NormalDistributionsFamily, m[:A]::PointMass, m[:in]::NormalDistributionsFamily),
    body = (ctx, args) -> promoted_cluster(
        FactorizedCluster((:A,) => args.m[:A], (:in,) => prod(ClosedProd(), unscaled(ctx, args.m[:out], mean(args.m[:A])), args.m[:in])),
        args.m[:out], args.m[:A], args.m[:in],
    ),
)

# A known scalar or vector `in` for a univariate A. With a vector `in`, any multivariate
# Gaussian `out` is accepted, as its message rule does.
@define_marginal_update_rule(
    node = *, target = (:A, :in), ctx = (:matrix_correction,),
    args = (m[:out]::NormalDistributionsFamily, m[:A]::UnivariateNormalDistributionsFamily, m[:in]::PointMass{<:Union{Real, AbstractVector}}),
    body = (ctx, args) -> promoted_cluster(
        FactorizedCluster((:A,) => prod(ClosedProd(), args.m[:A], unscaled(ctx, args.m[:out], mean(args.m[:in]))), (:in,) => args.m[:in]),
        args.m[:out], args.m[:A], args.m[:in],
    ),
)

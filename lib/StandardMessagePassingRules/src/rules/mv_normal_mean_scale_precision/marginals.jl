# As MvNormalMeanPrecision's, with the precision E[γ]·I coupling `out` and `μ`.
scale_precision(γ, d, ::Type{T}) where {T} = γ * diageye(T, d)

@define_marginal_update_rule(
    node = MvNormalMeanScalePrecision, target = (:out, :μ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::MultivariateNormalDistributionsFamily, q[:γ]::Any),
    body = (args) -> mv_coupled(args.m[:out], args.m[:μ], scale_precision(mean(args.q[:γ]), ndims(args.m[:out]), eltype(mean(args.m[:out])))),
)

@define_marginal_update_rule(
    node = MvNormalMeanScalePrecision, target = (:out, :μ),
    args = (m[:out]::PointMass, m[:μ]::MultivariateNormalDistributionsFamily, q[:γ]::Any),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => args.m[:out],
            (:μ,) => prod(
                ClosedProd(), MvNormalMeanPrecision(mean(args.m[:out]), scale_precision(mean(args.q[:γ]), ndims(args.m[:μ]), eltype(mean(args.m[:μ])))), args.m[:μ],
            ),
        ),
        args.m[:out], args.m[:μ], args.q[:γ],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanScalePrecision, target = (:out, :μ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::PointMass, q[:γ]::Any),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(
                ClosedProd(), MvNormalMeanPrecision(mean(args.m[:μ]), scale_precision(mean(args.q[:γ]), ndims(args.m[:out]), eltype(mean(args.m[:out])))), args.m[:out],
            ),
            (:μ,) => args.m[:μ],
        ),
        args.m[:out], args.m[:μ], args.q[:γ],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanScalePrecision, target = (:out, :μ, :γ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::PointMass, m[:γ]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(
                ClosedProd(), MvNormalMeanPrecision(mean(args.m[:μ]), scale_precision(mean(args.m[:γ]), ndims(args.m[:out]), eltype(mean(args.m[:out])))), args.m[:out],
            ),
            (:μ,) => args.m[:μ],
            (:γ,) => args.m[:γ],
        ),
        args.m[:out], args.m[:μ], args.m[:γ],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanScalePrecision, target = (:out, :μ, :γ),
    args = (m[:out]::PointMass, m[:μ]::MultivariateNormalDistributionsFamily, m[:γ]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => args.m[:out],
            (:μ,) => prod(
                ClosedProd(), args.m[:μ], MvNormalMeanPrecision(mean(args.m[:out]), scale_precision(mean(args.m[:γ]), ndims(args.m[:μ]), eltype(mean(args.m[:μ])))),
            ),
            (:γ,) => args.m[:γ],
        ),
        args.m[:out], args.m[:μ], args.m[:γ],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanScalePrecision, target = (:out, :μ, :γ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::MultivariateNormalDistributionsFamily, m[:γ]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out, :μ) => mv_coupled(args.m[:out], args.m[:μ], scale_precision(mean(args.m[:γ]), ndims(args.m[:out]), eltype(mean(args.m[:out])))),
            (:γ,) => args.m[:γ],
        ),
        args.m[:out], args.m[:μ], args.m[:γ],
    ),
)

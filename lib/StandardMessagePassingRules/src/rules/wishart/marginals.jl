# The whole node under belief propagation, with known ν and S: the message towards `out`
# times the prior, as a `Wishart`.
@define_marginal_update_rule(
    node = Wishart, target = (:out, :ν, :S),
    args = (m[:out]::WishartDistributionsFamily, m[:ν]::PointMass, m[:S]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => public_equivalent(prod(ClosedProd(), WishartFast(mean(args.m[:ν]), cholinv(mean(args.m[:S]))), args.m[:out])),
            (:ν,) => args.m[:ν],
            (:S,) => args.m[:S],
        ),
        args.m[:out], args.m[:ν], args.m[:S],
    ),
)

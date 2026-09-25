# The whole node under belief propagation, with known ν and S: the message towards `out`
# times the prior, as an `InverseWishart`. A message of any of the family is accepted, as
# Wishart's rule does.
@define_marginal_update_rule(
    node = InverseWishart, target = (:out, :ν, :S),
    args = (m[:out]::InverseWishartDistributionsFamily, m[:ν]::PointMass, m[:S]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => public_equivalent(prod(ClosedProd(), InverseWishartFast(mean(args.m[:ν]), mean(args.m[:S])), args.m[:out])),
            (:ν,) => args.m[:ν],
            (:S,) => args.m[:S],
        ),
        args.m[:out], args.m[:ν], args.m[:S],
    ),
)

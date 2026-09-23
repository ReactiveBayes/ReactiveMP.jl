# The whole node under belief propagation, with a known `a`: the message towards `out` times
# the prior.
@define_marginal_update_rule(
    node = DirichletCollection, target = (:out, :a),
    args = (m[:out]::DirichletCollection, m[:a]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster((:out,) => prod(ClosedProd(), DirichletCollection(mean(args.m[:a])), args.m[:out]), (:a,) => args.m[:a]),
        args.m[:out], args.m[:a],
    ),
)

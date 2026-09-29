@define_marginal_update_rule(
    node = Dirichlet, target = (:out, :a),
    args = (m[:out]::Dirichlet, m[:a]::PointMass),
    body = (args) -> promote_cluster(
        FactorizedCluster((:out,) => prod(ClosedProd(), Dirichlet(mean(args.m[:a])), args.m[:out]), (:a,) => args.m[:a]),
        args.m[:out], args.m[:a],
    ),
)

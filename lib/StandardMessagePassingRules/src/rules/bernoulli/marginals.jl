@define_marginal_update_rule(
    node = Bernoulli, target = (:out, :p),
    args = (m[:out]::PointMass, m[:p]::Beta),
    body = (args) -> promoted_cluster(
        FactorizedCluster((:out,) => args.m[:out], (:p,) => prod(ClosedProd(), bernoulli_likelihood(mean(args.m[:out])), args.m[:p])),
        args.m[:out], args.m[:p],
    ),
)

@define_marginal_update_rule(
    node = Bernoulli, target = (:out, :p),
    args = (m[:out]::Bernoulli, m[:p]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster((:out,) => prod(ClosedProd(), Bernoulli(mean(args.m[:p])), args.m[:out]), (:p,) => args.m[:p]),
        args.m[:out], args.m[:p],
    ),
)

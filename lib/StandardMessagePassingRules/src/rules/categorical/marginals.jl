@define_marginal_update_rule(
    node = Categorical, target = (:out, :p),
    args = (m[:out]::Categorical, m[:p]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster((:out,) => prod(ClosedProd(), Categorical(mean(args.m[:p])), args.m[:out]), (:p,) => args.m[:p]),
        args.m[:out], args.m[:p],
    ),
)

@define_marginal_update_rule(
    node = Categorical, target = (:out, :p),
    args = (m[:out]::PointMass, m[:p]::Dirichlet),
    body = (args) -> begin
        probs = probvec(args.m[:out])
        p = prod(ClosedProd(), Dirichlet(probs .+ one(eltype(probs))), args.m[:p])
        promoted_cluster(FactorizedCluster((:out,) => args.m[:out], (:p,) => p), args.m[:out], args.m[:p])
    end,
)

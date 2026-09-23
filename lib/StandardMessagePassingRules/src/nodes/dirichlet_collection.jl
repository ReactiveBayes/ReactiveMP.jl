@define_factor_node(node = DirichletCollection, type = Stochastic, interfaces = [:out, :a])

# A collection of Dirichlets along the first dimension: the energy is the sum of each one's,
# -log Γ(Σ a) + Σ log Γ(a) - Σ (a - 1) E[log out], for a known `a`.
@define_average_energy(
    node = DirichletCollection,
    args = (q[:out]::DirichletCollection, q[:a]::PointMass),
    body = (args) -> begin
        a = mean(args.q[:a])
        sum(-loggamma.(sum(a, dims = 1)) .+ sum(loggamma.(a), dims = 1) .- sum((a .- 1) .* mean(BroadcastFunction(log), args.q[:out]), dims = 1))
    end,
)

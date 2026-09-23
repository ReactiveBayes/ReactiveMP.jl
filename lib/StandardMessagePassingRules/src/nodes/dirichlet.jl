@define_factor_node(node = Dirichlet, type = Stochastic, interfaces = [:out, :a], algorithm = BP)

@define_average_energy(
    node = Dirichlet,
    args = (q[:out]::Dirichlet, q[:a]::PointMass),
    body = (args) -> begin
        a = mean(args.q[:a])
        -loggamma(sum(a)) + sum(loggamma.(a)) - sum((a .- 1) .* mean(BroadcastFunction(log), args.q[:out]))
    end,
)

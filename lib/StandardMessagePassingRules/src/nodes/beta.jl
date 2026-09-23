@define_factor_node(node = Beta, type = Stochastic, interfaces = [:out, (:a, aliases = [:α]), (:b, aliases = [:β])])

@define_average_energy(
    node = Beta,
    args = (q[:out]::Any, q[:a]::Any, q[:b]::Any),
    body = (args) -> begin
        a, b = mean(args.q[:a]), mean(args.q[:b])
        logbeta(a, b) - (a - 1) * mean(log, args.q[:out]) - (b - 1) * mean(mirrorlog, args.q[:out])
    end,
)

@define_factor_node(node = Categorical, type = Stochastic, interfaces = [:out, :p])

@define_average_energy(
    node = Categorical,
    args = (q[:out]::Union{Categorical, Multinomial}, q[:p]::Any),
    body = (args) -> -sum(probvec(args.q[:out]) .* mean(BroadcastFunction(clamplog), args.q[:p])),
)

@define_message_update_rule(
    node = Categorical, target = :out,
    args = (m[:p]::Dirichlet,),
    logscale = 0,
    body = (args) -> begin
        Categorical(mean(args.m[:p]))
    end,
)

@define_message_update_rule(
    node = Categorical, target = :out,
    args = (q[:p]::Dirichlet,),
    body = (args) -> begin
        # Softened, so that no category ever gets exactly zero probability. `max` with `tiny`
        # keeps the input's precision, where a clamp to `[tiny, Inf]` would turn every input
        # into `Float64` through its `Inf` bound.
        rho = max.(exp.(mean(BroadcastFunction(log), args.q[:p])), tiny)
        Categorical(rho ./ sum(rho))
    end,
)

@define_message_update_rule(
    node = Categorical, target = :out,
    args = (m[:p]::PointMass,),
    logscale = 0,
    body = (args) -> Categorical(mean(args.m[:p])),
)

@define_message_update_rule(
    node = Categorical, target = :out,
    args = (q[:p]::PointMass,),
    logscale = 0,
    body = (args) -> Categorical(mean(args.q[:p])),
)

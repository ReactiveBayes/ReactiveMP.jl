@define_message_update_rule(
    node = Categorical, target = :out,
    args = (m[:p]::Dirichlet,),
    body = (args, ann) -> begin
        annotate!(ann, :logscale, 0)
        Categorical(mean(args.m[:p]))
    end,
)

@define_message_update_rule(
    node = Categorical, target = :out,
    args = (q[:p]::Dirichlet,),
    body = (args) -> begin
        # Softened, so that no category ever gets exactly zero probability. v6 clamped to
        # `[tiny, Inf]`, and the `Inf` bound turned every input into `Float64`; `max` with
        # `tiny` keeps the input's precision, and gives the same numbers.
        rho = max.(exp.(mean(BroadcastFunction(log), args.q[:p])), tiny)
        Categorical(rho ./ sum(rho))
    end,
)

@define_message_update_rule(
    node = Categorical, target = :out,
    args = (m[:p]::PointMass,),
    body = (args) -> Categorical(mean(args.m[:p])),
)

@define_message_update_rule(
    node = Categorical, target = :out,
    args = (q[:p]::PointMass,),
    body = (args) -> Categorical(mean(args.q[:p])),
)

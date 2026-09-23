@define_message_update_rule(
    node = NormalMeanPrecision, target = :τ,
    args = (q[:out]::Any, q[:μ]::Any),
    body = (args) -> begin
        θ = 2 / (var(args.q[:out]) + var(args.q[:μ]) + abs2(mean(args.q[:out]) - mean(args.q[:μ])))
        Gamma(convert(typeof(θ), 1.5), θ)
    end,
)

@define_message_update_rule(
    node = NormalMeanPrecision, target = :τ,
    args = (q[:out, :μ]::Any,),
    body = (args) -> begin
        m, V = mean_cov(args.q[:out, :μ])
        θ = 2 / (V[1, 1] - V[1, 2] - V[2, 1] + V[2, 2] + abs2(m[1] - m[2]))
        Gamma(convert(typeof(θ), 1.5), θ)
    end,
)

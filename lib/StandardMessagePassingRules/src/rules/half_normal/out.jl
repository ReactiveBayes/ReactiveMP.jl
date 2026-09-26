@define_message_update_rule(
    node = HalfNormal, target = :out,
    args = (q[:v]::PointMass,),
    logscale = 0,
    body = (args) -> begin
        v = mean(args.q[:v])
        truncated(Normal(zero(eltype(args.q[:v])), sqrt(v)), zero(eltype(args.q[:v])), typemax(float(v)))
    end,
)

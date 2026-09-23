@define_message_update_rule(
    node = NormalMixture, target = (:p, k),
    args = (q[:out]::Any, q[:switch]::Any, q[:m][k]::Any),
    body = (args) -> begin
        m_mean, m_var = mean_var(args.q[:m][k])
        out_mean, out_var = mean_var(args.q[:out])
        z = probvec(args.q[:switch])[k]
        GammaShapeRate(one(z) + z / 2, z * (out_var + m_var + abs2(out_mean - m_mean)) / 2)
    end,
)

@define_message_update_rule(
    node = NormalMixture, target = (:p, k),
    args = (q[:out]::Any, q[:switch]::PointMass{<:Real}, q[:m][k]::Any),
    body = (args) -> error(INTEGER_SWITCH),
)

"""
    HalfNormal

The half-normal node, `out` distributed as |N(0, v)|, with `v` a variance.
"""
struct HalfNormal end

@define_factor_node(node = HalfNormal, type = Stochastic, interfaces = [:out, (:v, aliases = [:var, :σ²])])

@define_average_energy(
    node = HalfNormal,
    args = (q[:out]::Any, q[:v]::Any),
    body = (args) -> begin
        out_mean, out_var = mean_var(args.q[:out])
        (log(π / 2) + mean(log, args.q[:v]) + mean(inv, args.q[:v]) * (out_mean^2 + out_var)) / 2
    end,
)

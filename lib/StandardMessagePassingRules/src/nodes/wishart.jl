@define_factor_node(node = Wishart, type = Stochastic, interfaces = [:out, (:ν, aliases = [:df]), (:S, aliases = [:scale])])

# The Wishart rules work in `WishartFast`, which stores the inverse scale; a marginal is
# formed as Distributions' `Wishart`. InverseWishart's fast form is converted the same way.
MessagePassingRulesBase.public_equivalent(d::WishartFast) = convert(Wishart, d)
MessagePassingRulesBase.public_equivalent(d::InverseWishartFast) = convert(InverseWishart, d)

# -E[log Wishart(out | ν, S)] for a known ν: (ν (E[log |S|] + d log 2) - (ν - d - 1) E[log |out|]
# + tr(E[S⁻¹] E[out]) + d(d - 1)/2 log π) / 2 + Σᵢ log Γ((ν + 1 - i)/2). The constants are
# taken in ν's float type.
function wishart_energy(q_out, ν, q_S)
    d = size(q_out, 1)
    ν = float(ν)
    log2, logpi = log(oftype(ν, 2)), oftype(ν, logπ)
    return (
        ν * (mean(logdet, q_S) + d * log2) - mean(logdet, q_out) * (ν - d - 1) + tr(mean(cholinv, q_S) * mean(q_out)) +
            d * (d - 1) * logpi / 2
    ) / 2 + sum(i -> loggamma((ν + 1 - i) / 2), 1:d)
end

@define_average_energy(
    node = Wishart,
    args = (q[:out]::Any, q[:ν]::PointMass, q[:S]::Any),
    body = (args) -> wishart_energy(args.q[:out], mean(args.q[:ν]), args.q[:S]),
)

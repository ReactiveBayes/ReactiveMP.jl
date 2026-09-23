@define_factor_node(node = InverseWishart, type = Stochastic, interfaces = [:out, (:ν, aliases = [:df]), (:S, aliases = [:scale, :Ψ])])

# -E[log InverseWishart(out | ν, S)] for a known ν: (ν (d log 2 - E[log |S|]) + (ν + d + 1)
# E[log |out|] + tr(E[S] E[out⁻¹]) + d(d - 1)/2 log π) / 2 + Σᵢ log Γ((ν + 1 - i)/2). The
# constants are taken in ν's float type.
function inverse_wishart_energy(q_out, ν, q_S)
    d = size(q_out, 1)
    ν = float(ν)
    log2, logpi = log(oftype(ν, 2)), oftype(ν, logπ)
    return (
        ν * (d * log2 - mean(logdet, q_S)) + mean(logdet, q_out) * (ν + d + 1) + tr(mean(q_S) * mean(cholinv, q_out)) +
            d * (d - 1) * logpi / 2
    ) / 2 + sum(i -> loggamma((ν + 1 - i) / 2), 1:d)
end

@define_average_energy(
    node = InverseWishart,
    args = (q[:out]::Any, q[:ν]::PointMass, q[:S]::Any),
    body = (args) -> inverse_wishart_energy(args.q[:out], mean(args.q[:ν]), args.q[:S]),
)

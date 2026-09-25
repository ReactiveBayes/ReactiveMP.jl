@define_factor_node(
    node = MatrixNormalWishart, type = Stochastic,
    interfaces = [:out, (:M, aliases = [:mean]), (:U, aliases = [:rowcov]), (:V, aliases = [:scale]), (:ν, aliases = [:dof])],
)

# -E[log MatrixNormalWishart((X, Y) | M, U, V, ν)] for known parameters, where
# X | Y ~ MatrixNormal(M, U, Y⁻¹) and Y ~ Wishart(ν, V), under q(out) = MatrixNormalWishart(Mq,
# Uq, Vq, νq): with E[Y] = νq Vq, L = E[log |Y|] and D = Mq - M,
#   n p/2 log 2π + p/2 log |U| + ν p/2 log 2 + ν/2 log |V| + log Γ_p(ν/2) - (n + ν - p - 1)/2 L
#   + (tr(U⁻¹ D E[Y] Dᵀ) + p tr(U⁻¹ Uq) + tr(V⁻¹ E[Y])) / 2.
# The parameters are known.
function matrix_normal_wishart_energy(q_out, M, U, V, ν)
    Mq, Uq, Vq, νq = params(q_out)
    n, p = size(Mq)
    ν = float(ν)
    log2, logpi = log(oftype(ν, 2)), oftype(ν, logπ)
    L = p * log2 + logdet(Vq) + sum(i -> digamma((νq + 1 - i) / 2), 1:p)
    logΓp = p * (p - 1) * logpi / 4 + sum(i -> loggamma((ν + 1 - i) / 2), 1:p)
    invU, EY = cholinv(U), νq * Vq
    D = Mq - M
    rest = p * logdet(U) + ν * p * log2 + ν * logdet(V) + 2 * logΓp - (n + ν - p - 1) * L +
        dot(invU', D * EY * D') + p * dot(invU', Uq) + dot(cholinv(V)', EY)
    return gaussian_energy(n * p, rest)
end

@define_average_energy(
    node = MatrixNormalWishart,
    args = (q[:out]::MatrixNormalWishart, q[:M]::PointMass, q[:U]::PointMass, q[:V]::PointMass, q[:ν]::PointMass),
    body = (args) -> matrix_normal_wishart_energy(args.q[:out], mean(args.q[:M]), mean(args.q[:U]), mean(args.q[:V]), mean(args.q[:ν])),
)

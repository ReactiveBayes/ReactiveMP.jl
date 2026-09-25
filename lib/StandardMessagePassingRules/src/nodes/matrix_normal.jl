@define_factor_node(
    node = MatrixNormal, type = Stochastic,
    interfaces = [:out, (:M, aliases = [:mean]), (:U, aliases = [:rowcov]), (:V, aliases = [:colcov])],
)

# A matrix `out` or `M` is known (a point mass) or matrix normal; the rules and the energy need
# its second moments, and no other type has them in this form.
const PointOrMatrixNormal = Union{PointMass, MatrixNormal}

# E[(out - M) B (out - M)ᵀ] = D B Dᵀ + tr(B V_out) U_out + tr(B V_M) U_M for D = E[out] - E[M],
# using E[X B Xᵀ] = E[X] B E[X]ᵀ + tr(B V) U for X ~ MatrixNormal(·, U, V).
row_scatter(q_out, q_M, B) = (D = mean(q_out) - mean(q_M); row_moment(q_M, B, row_moment(q_out, B, D * B * D')))
row_moment(::PointMass, B, Ψ) = Ψ
row_moment(q::MatrixNormal, B, Ψ) = ((U, V) = covmats(q); Ψ + dot(B', V) * U)

# E[(out - M)ᵀ A (out - M)] = Dᵀ A D + tr(A U_out) V_out + tr(A U_M) V_M, by the same identity.
column_scatter(q_out, q_M, A) = (D = mean(q_out) - mean(q_M); column_moment(q_M, A, column_moment(q_out, A, D' * A * D)))
column_moment(::PointMass, A, Ψ) = Ψ
column_moment(q::MatrixNormal, A, Ψ) = ((U, V) = covmats(q); Ψ + dot(A', U) * V)

# (n p log 2π + p E[log |U|] + n E[log |V|] + tr(E[U⁻¹] E[(out - M) E[V⁻¹] (out - M)ᵀ])) / 2.
# `q_out` and `q_M` are narrowed to the types whose second moments are known.
@define_average_energy(
    node = MatrixNormal,
    args = (q[:out]::PointOrMatrixNormal, q[:M]::PointOrMatrixNormal, q[:U]::Any, q[:V]::Any),
    body = (args) -> begin
        n, p = size(mean(args.q[:out]))
        Ψ = row_scatter(args.q[:out], args.q[:M], mean(cholinv, args.q[:V]))
        gaussian_energy(n * p, p * mean(logdet, args.q[:U]) + n * mean(logdet, args.q[:V]) + dot(mean(cholinv, args.q[:U])', Ψ))
    end,
)

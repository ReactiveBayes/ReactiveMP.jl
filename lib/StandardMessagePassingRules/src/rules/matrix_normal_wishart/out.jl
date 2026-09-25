# Known parameters only. ExponentialFamily's constructor promotes the matrices but
# keeps ν's own float type; a message has one.
function promoted_matrix_normal_wishart(M, U, V, ν)
    T = promote_type(eltype(M), eltype(U), eltype(V), typeof(ν))
    return MatrixNormalWishart(convert(AbstractMatrix{T}, M), convert(AbstractMatrix{T}, U), convert(AbstractMatrix{T}, V), convert(T, ν))
end

@define_message_update_rule(
    node = MatrixNormalWishart, target = :out,
    args = (q[:M]::PointMass, q[:U]::PointMass, q[:V]::PointMass, q[:ν]::PointMass),
    body = (args) -> promoted_matrix_normal_wishart(mean(args.q[:M]), mean(args.q[:U]), mean(args.q[:V]), mean(args.q[:ν])),
)

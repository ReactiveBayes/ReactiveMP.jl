# Whether `vec` has exactly one entry approximately equal to one and all others approximately
# zero, to within `sqrt(eps)` of its element type.
function isonehot(vec::AbstractVector{T}) where {T}
    ones_seen = 0
    atol = sqrt(eps(T))
    for e in vec
        if isapprox(e, one(e); atol)
            ones_seen += 1
        elseif !isapprox(e, zero(e); atol)
            return false
        end
    end
    return ones_seen == 1
end

# `cluster` with every block in the float type of all `inputs` together. A rule's output must
# carry the promoted float type of every input, and that includes a block that passes an input
# through unchanged, such as v6's `v = m_v`.
promoted_cluster(cluster::FactorizedCluster, inputs...) =
    BayesBase.convert_paramfloattype(BayesBase.promote_paramfloattype(inputs...), cluster)

"""
    diageye([T = Float64], n)

The `n`×`n` identity matrix of element type `T`, dense.
"""
diageye(::Type{T}, n::Integer) where {T} = Matrix{T}(I, n, n)
diageye(n::Integer) = diageye(Float64, n)

"""
    negate_inplace!(A)

`-A`, overwriting `A` when it is an `Array`; any other array, and a number, is left as it is.
"""
function negate_inplace! end

negate_inplace!(A::AbstractArray) = -A
negate_inplace!(A::Real) = -A
negate_inplace!(A::Array) = map!(-, A, A)

"""
    mul_inplace!(alpha, A)

`alpha * A`, overwriting `A` when it is an `Array` of `alpha`'s type; any other array, and a
number, is left as it is.
"""
function mul_inplace! end

mul_inplace!(alpha, A::AbstractArray) = alpha * A
mul_inplace!(alpha, A::Real) = alpha * A
mul_inplace!(alpha::T, A::Array{T}) where {T <: Real} = LinearAlgebra.lmul!(alpha, A)

"""
    rank1update(A, x)
    rank1update(A, x, y)

`A + x * y'` (`y = x` when omitted) in a new matrix, by BLAS for dense matrices of one BLAS
float type, and by a loop otherwise, as for dual numbers.
"""
function rank1update end

rank1update(A::AbstractMatrix, x::AbstractVector) = rank1update(eltype(A), eltype(x), eltype(x), A, x, x)
rank1update(A::AbstractMatrix, x::AbstractVector, y::AbstractVector) = rank1update(eltype(A), eltype(x), eltype(y), A, x, y)
rank1update(A::Real, x::Real) = rank1update(A, x, x)
rank1update(A::Real, x::Real, y::Real) = A + x * y

rank1update(::Type{T}, ::Type{T}, ::Type{T}, A::Matrix, x::Vector, y::Vector) where {T <: LinearAlgebra.BlasFloat} =
    LinearAlgebra.BLAS.ger!(one(T), x, y, copy(A))

function rank1update(::Type{T1}, ::Type{T2}, ::Type{T3}, A::AbstractMatrix, x::AbstractVector, y::AbstractVector) where {T1 <: Real, T2 <: Real, T3 <: Real}
    B = Matrix{promote_type(T1, T2, T3)}(undef, size(A))
    @inbounds for k2 in axes(A, 2), k1 in axes(A, 1)
        B[k1, k2] = A[k1, k2] + x[k1] * y[k2]
    end
    return B
end

"""
    mul_trace(A, B)

`tr(A * B)`, without forming the product.
"""
function mul_trace end

mul_trace(A::Real, B::Real) = A * B

function mul_trace(A::AbstractMatrix, B::AbstractMatrix)
    n = LinearAlgebra.checksquare(A)
    size(B) == size(A) || throw(DimensionMismatch("mul_trace: sizes $(size(A)) and $(size(B)) differ"))
    result = zero(promote_type(eltype(A), eltype(B)))
    @inbounds for i in 1:n, j in 1:n
        result += A[i, j] * B[j, i]
    end
    return result
end

# The precision of the joint of `out` and `μ` when `out` is normal around `μ` with precision
# `W_bar` and each carries a message of precision `W_out`, `W_μ`:
# `[W_out + W_bar  -W_bar; -W_bar  W_μ + W_bar]`, for scalars and matrices alike.
coupled_precision(W_out, W_μ, W_bar) = [W_out + W_bar -W_bar; -W_bar W_μ + W_bar]

# E[(out - μ)(out - μ)ᵀ], for independent marginals of `out` and `μ`, or for their joint, whose
# first half is `out` and second half `μ`.
function difference_moment(q_out, q_μ)
    m_out, V_out = mean_cov(q_out)
    m_μ, V_μ = mean_cov(q_μ)
    Δ = m_out - m_μ
    return V_out + V_μ + Δ * Δ'
end

function difference_moment(q_joint)
    m, V = mean_cov(q_joint)
    d = div(length(m), 2)
    Δ = @views m[1:d] - m[(d + 1):end]
    return @views V[1:d, 1:d] - V[1:d, (d + 1):end] - V[(d + 1):end, 1:d] + V[(d + 1):end, (d + 1):end] + Δ * Δ'
end

# The covariance a marginal `q_Σ` of a covariance contributes under naive VMP, E[Σ⁻¹]⁻¹; v6 used
# E[Σ] (ReactiveMP.jl#673, the multivariate form of #669). They agree for a point mass.
variational_covariance(q_Σ) = cholinv(mean(cholinv, q_Σ))

# The Gaussian message `m` through Gaussian noise of precision `Λ_f`: the mean of `m` and the
# precision (Λ⁻¹ + Λ_f⁻¹)⁻¹, computed as Λ - Λ (Λ + Λ_f)⁻¹ Λ with one Cholesky factorisation.
function series_precision(m, Λ_f)
    μ, Λ = mean_precision(m)
    return MvNormalMeanPrecision(μ, Λ - Λ * (fastcholesky(Λ + Λ_f) \ Λ))
end

# (d log 2π + rest) / 2, the average energy of a d-dimensional normal given the rest of it, in
# `rest`'s float type: `d * log2π` alone is a Float64 for an `Int` `d`, whatever the inputs.
gaussian_energy(d, rest) = (d * oftype(rest, log2π) + rest) / 2

# The math helpers the rule packages share: linear algebra and the Gaussian algebra of rules.
# Public and documented (the page Math helpers), not exported.

"""
    add_outer(A, x)
    add_outer(A, x, y)

Return `A + x * y'`, with `y = x` when omitted, in a new matrix; `A` is not modified. Dense
arguments of one BLAS float type go through BLAS `ger!`; any other element types, dual numbers
say, through a loop in their promoted type. For numbers it is `A + x * y`.

# Examples

```jldoctest
julia> MessagePassingRulesBase.add_outer([1.0 0.0; 0.0 1.0], [1.0, 2.0])
2×2 Matrix{Float64}:
 2.0  2.0
 2.0  5.0

julia> MessagePassingRulesBase.add_outer(1.0, 2.0, 3.0)
7.0
```

See also [`gaussian_second_moment`](@ref), [`gaussian_cross_moment`](@ref).
"""
function add_outer end

add_outer(A::AbstractMatrix, x::AbstractVector) = add_outer(eltype(A), eltype(x), eltype(x), A, x, x)
add_outer(A::AbstractMatrix, x::AbstractVector, y::AbstractVector) = add_outer(eltype(A), eltype(x), eltype(y), A, x, y)
add_outer(A::Real, x::Real) = add_outer(A, x, x)
add_outer(A::Real, x::Real, y::Real) = A + x * y

add_outer(::Type{T}, ::Type{T}, ::Type{T}, A::Matrix, x::Vector, y::Vector) where {T <: LinearAlgebra.BlasFloat} =
    LinearAlgebra.BLAS.ger!(one(T), x, y, copy(A))

function add_outer(::Type{T1}, ::Type{T2}, ::Type{T3}, A::AbstractMatrix, x::AbstractVector, y::AbstractVector) where {T1 <: Real, T2 <: Real, T3 <: Real}
    B = Matrix{promote_type(T1, T2, T3)}(undef, size(A))
    @inbounds for k2 in axes(A, 2), k1 in axes(A, 1)
        B[k1, k2] = A[k1, k2] + x[k1] * y[k2]
    end
    return B
end

"""
    trace_product(A, B)

The trace `tr(A * B)`, computed without forming the product; for numbers, `A * B`.

# Throws

- `DimensionMismatch` when `A` is not square or `B` has another size.

# Examples

```jldoctest
julia> MessagePassingRulesBase.trace_product([1.0 2.0; 3.0 4.0], [1.0 0.0; 0.0 1.0])
5.0
```
"""
function trace_product end

trace_product(A::Real, B::Real) = A * B

function trace_product(A::AbstractMatrix, B::AbstractMatrix)
    n = LinearAlgebra.checksquare(A)
    size(B) == size(A) || throw(DimensionMismatch("trace_product: sizes $(size(A)) and $(size(B)) differ"))
    result = zero(promote_type(eltype(A), eltype(B)))
    @inbounds for i in 1:n, j in 1:n
        result += A[i, j] * B[j, i]
    end
    return result
end

"""
    negate!!(A)

Return `-A`, overwriting `A` when it is a dense `Array`; any other array, a view say, and a
number are left as they are, and a new value is returned. The `!!` says it mutates where it can,
so it is given a value the caller owns, such as a fresh product: `negate!!(W * A)`.

See also [`scale!!`](@ref).
"""
function negate!! end

negate!!(A::AbstractArray) = -A
negate!!(A::Real) = -A
negate!!(A::Array) = map!(-, A, A)

"""
    scale!!(α, A)

Return `α * A`, overwriting `A` when it is a dense `Array` whose element type is `α`'s real
type; any other array and a number are left as they are, and a new value is returned. As for
[`negate!!`](@ref), `A` is a value the caller owns.
"""
function scale!! end

scale!!(α, A::AbstractArray) = α * A
scale!!(α, A::Real) = α * A
scale!!(α::T, A::Array{T}) where {T <: Real} = LinearAlgebra.lmul!(α, A)

"""
    scaled_outer(v, a)

The product `v a vᵀ`. For a vector `v` and a real `a` it is computed as `(v vᵀ) a`, which is
exactly symmetric where `(v a) vᵀ` is not always; for anything else, as `v * a * v'`.

A package with a structured vector adds a method for it: the autoregressive package gives its
standard basis vector `e` the diagonal `e a eᵀ`, which the rules of `dot` then build.

# Examples

```jldoctest
julia> MessagePassingRulesBase.scaled_outer([1.0, 2.0], 3.0)
2×2 Matrix{Float64}:
 3.0   6.0
 6.0  12.0
```
"""
function scaled_outer end

scaled_outer(v, a) = v * a * v'
scaled_outer(v::AbstractVector, a::Real) = v * v' * a

"""
    diageye([T = Float64,] n::Integer) -> Matrix{T}

The `n`×`n` identity matrix of element type `T`, as a dense `Matrix`, for the covariances and
precisions a model writes, `MvNormalMeanCovariance(zeros(2), diageye(2))`. Unlike
`LinearAlgebra.I`, it has a size and can be inverted, factorised and mutated.

# Examples

```jldoctest
julia> MessagePassingRulesBase.diageye(2)
2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0

julia> MessagePassingRulesBase.diageye(Int, 1)
1×1 Matrix{Int64}:
 1
```
"""
diageye(::Type{T}, n::Integer) where {T} = Matrix{T}(LinearAlgebra.I, n, n)
diageye(n::Integer) = diageye(Float64, n)

"""
    promote_cluster(cluster::FactorizedCluster, inputs...) -> FactorizedCluster

`cluster` with every block in the float type of all `inputs` together, as
`BayesBase.promote_paramfloattype` gives it. A rule's output carries the promoted float type of
every input, and that includes a block that passes an input through unchanged.

See also [`FactorizedCluster`](@ref).
"""
promote_cluster(cluster::FactorizedCluster, inputs...) =
    BayesBase.convert_paramfloattype(BayesBase.promote_paramfloattype(inputs...), cluster)

"""
    gaussian_average_energy(d, rest)

`(d log 2π + rest) / 2`, the average energy of a `d`-dimensional normal given `rest`, the
remainder of its expectation, such as `E[(x - μ)ᵀ Λ (x - μ)] - E[log det Λ]`. It is in `rest`'s
float type: `d * log2π` alone is a `Float64` for an integer `d`, whatever the inputs.

# Examples

```jldoctest
julia> MessagePassingRulesBase.gaussian_average_energy(1, 0.0f0) isa Float32
true
```
"""
gaussian_average_energy(d, rest) = (d * oftype(rest, log2π) + rest) / 2

"""
    gaussian_variational_variance(q_v)

The variance a normal factor sees of a marginal `q_v` over its variance under naive variational
message passing, `E[v⁻¹]⁻¹`: the variance whose precision is the expected precision. For a point
mass it is the variance itself.

See also [`gaussian_variational_covariance`](@ref).
"""
gaussian_variational_variance(q_v) = inv(BayesBase.mean(inv, q_v))

"""
    gaussian_variational_covariance(q_Σ)

The covariance a multivariate normal factor sees of a marginal `q_Σ` over its covariance under
naive variational message passing, `E[Σ⁻¹]⁻¹`, inverted by Cholesky. For a point mass it is `Σ`.

See also [`gaussian_variational_variance`](@ref).
"""
gaussian_variational_covariance(q_Σ) = cholinv(BayesBase.mean(cholinv, q_Σ))

"""
    gaussian_coupled_precision(W_out, W_μ, W_bar)

The precision of the joint of `out` and `μ` when `out` is normal around `μ` with precision
`W_bar`, and `out` and `μ` carry messages of precisions `W_out` and `W_μ`:
`[W_out + W_bar  -W_bar; -W_bar  W_μ + W_bar]`, for numbers and matrices alike.

# Examples

```jldoctest
julia> MessagePassingRulesBase.gaussian_coupled_precision(1.0, 2.0, 0.5)
2×2 Matrix{Float64}:
  1.5  -0.5
 -0.5   2.5
```
"""
gaussian_coupled_precision(W_out, W_μ, W_bar) = [W_out + W_bar -W_bar; -W_bar W_μ + W_bar]

"""
    gaussian_difference_moment(q_out, q_μ)
    gaussian_difference_moment(q_joint)

`E[(out - μ)(out - μ)ᵀ]`: for independent marginals of `out` and `μ`, or for their joint, whose
first half is `out` and second half `μ`. Each marginal needs `mean_cov`.
"""
function gaussian_difference_moment(q_out, q_μ)
    m_out, V_out = BayesBase.mean_cov(q_out)
    m_μ, V_μ = BayesBase.mean_cov(q_μ)
    Δ = m_out - m_μ
    return V_out + V_μ + Δ * Δ'
end

function gaussian_difference_moment(q_joint)
    m, V = BayesBase.mean_cov(q_joint)
    d = div(length(m), 2)
    Δ = @views m[1:d] - m[(d + 1):end]
    return @views V[1:d, 1:d] - V[1:d, (d + 1):end] - V[(d + 1):end, 1:d] + V[(d + 1):end, (d + 1):end] + Δ * Δ'
end

"""
    gaussian_series_precision(Λ, Λ_f)

The precision of a normal of precision `Λ` passed through normal noise of precision `Λ_f`,
`(Λ⁻¹ + Λ_f⁻¹)⁻¹`, computed as `Λ - Λ (Λ + Λ_f)⁻¹ Λ` with one Cholesky factorisation. The mean is
unchanged.
"""
gaussian_series_precision(Λ, Λ_f) = Λ - Λ * (fastcholesky(Λ + Λ_f) \ Λ)

"""
    gaussian_second_moment(q)

`E[x xᵀ] = Cov[x] + E[x] E[x]ᵀ` of a multivariate `q`, and `E[x²] = Var[x] + E[x]²` of a
univariate one, from `mean_cov(q)`: any distribution or marginal that defines it.

See also [`add_outer`](@ref), [`gaussian_cross_moment`](@ref).
"""
function gaussian_second_moment(q)
    m, V = BayesBase.mean_cov(q)
    return add_outer(V, m)
end

"""
    gaussian_cross_moment(V_xy, m_x, m_y)

`E[x yᵀ] = Cov[x, y] + E[x] E[y]ᵀ`, from the cross-covariance of `x` and `y` in their joint and
their means; for numbers, `Cov[x, y] + E[x] E[y]`.

See also [`gaussian_second_moment`](@ref).
"""
gaussian_cross_moment(V_xy, m_x, m_y) = add_outer(V_xy, m_x, m_y)

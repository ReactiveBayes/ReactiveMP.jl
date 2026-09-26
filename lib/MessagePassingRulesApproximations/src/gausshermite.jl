"""
    GaussHermiteCubature(p::Int)

Gauss–Hermite cubature with `p` points per dimension, for expectations under a normal given by
its mean and variance, or mean vector and covariance. It is exact for polynomials of degree up to
`2p - 1` in each coordinate. In `d` dimensions it evaluates `p^d` points, a tensor product, so
it suits a handful of dimensions only.

It provides [`getpoints`](@ref) and [`getweights`](@ref), and through them
[`approximate_meancov`](@ref). [`ghcubature`](@ref)`(p)` builds the same.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> gh = GaussHermiteCubature(3);

julia> sum(w * x^2 for (w, x) in zip(getweights(gh, 1.0, 4.0), getpoints(gh, 1.0, 4.0))) ≈ 5.0  # E[x²] = m² + v
true
```
"""
struct GaussHermiteCubature{PI, WI} <: AbstractApproximationMethod
    piter::PI
    witer::WI
end

GaussHermiteCubature(p::Int) = ghcubature(p)

"""
    ghcubature(p::Int) -> GaussHermiteCubature

A [`GaussHermiteCubature`](@ref) with `p` points per dimension, its nodes and weights computed by
FastGaussQuadrature.
"""
function ghcubature(p::Int)
    points, weights = FastGaussQuadrature.gausshermite(p)
    return GaussHermiteCubature(points, weights)
end

approximation_name(gh::GaussHermiteCubature) = "GaussHermite($(length(gh.piter)))"
approximation_short_name(gh::GaussHermiteCubature) = "GH$(length(gh.piter))"

"""
    getweights(method, mean::Real, variance::Real)
    getweights(method, mean::AbstractVector, covariance::AbstractMatrix)

The cubature weights of `method` for a normal with these moments: a lazy generator of numbers
summing to one, in the order of [`getpoints`](@ref), so that `E[f(x)] ≈ Σ wᵢ f(xᵢ)`.
"""
function getweights end

"""
    getpoints(method, mean::Real, variance::Real)
    getpoints(method, mean::AbstractVector, covariance::AbstractMatrix)

The cubature points of `method` for a normal with these moments: a lazy generator of numbers, or
of vectors in the multivariate case, in the order of [`getweights`](@ref).

!!! warning
    The multivariate generator yields **one buffer**, rewritten on every iteration, so that a
    sum over the points allocates nothing. `collect` therefore returns copies of the last point;
    use `map(copy, getpoints(…))` to keep them. A consumer may overwrite the point it is given,
    as [`approximate_meancov`](@ref) does.
"""
function getpoints end

getweights(gh::GaussHermiteCubature, mean::T, variance::T) where {T <: Real} =
    Base.Generator(weight -> weight / sqrt(π), gh.witer)

function getweights(gh::GaussHermiteCubature, mean::AbstractVector{T}, covariance::AbstractMatrix{T}) where {T <: Real}
    normalisation = π^(length(mean) / 2)
    return Base.Generator(weights -> prod(weights) / normalisation, Iterators.product(Iterators.repeated(gh.witer, length(mean))...))
end

function getpoints(gh::GaussHermiteCubature, mean::T, variance::T) where {T <: Real}
    scale = sqrt(2 * variance)
    return Base.Generator(point -> mean + scale * point, gh.piter)
end

function getpoints(gh::GaussHermiteCubature, mean::AbstractVector{T}, covariance::AbstractMatrix{T}) where {T <: Real}
    sqrtP = cholsqrt(covariance)
    point, unit = similar(mean), similar(mean)
    return Base.Generator(Iterators.product(Iterators.repeated(gh.piter, length(mean))...)) do units
        copyto!(unit, units)
        copyto!(point, mean)
        return mul!(point, sqrtP, unit, sqrt(2), 1.0) # mean + √2 · sqrtP · unit
    end
end

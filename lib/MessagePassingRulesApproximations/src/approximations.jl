"""
    AbstractApproximationMethod

The supertype of the approximation methods: [`Unscented`](@ref), [`Linearization`](@ref) and
[`GaussHermiteCubature`](@ref). A method implements what its callers need: [`approximate`](@ref)
to propagate moments through a function, or [`getpoints`](@ref) and [`getweights`](@ref) for a
cubature, which gives it [`approximate_meancov`](@ref); and [`approximation_name`](@ref) and
[`approximation_short_name`](@ref) for display.
"""
abstract type AbstractApproximationMethod end

"""
    approximation_name(method::AbstractApproximationMethod)

The method's full name, for display: `"Unscented"`, `"Linearization"`, or
`"GaussHermite(p)"` with its number of points. See also [`approximation_short_name`](@ref).
"""
function approximation_name end

"""
    approximation_short_name(method::AbstractApproximationMethod)

The method's abbreviated name, for display: `"UT"`, `"LN"`, or `"GHp"` with its number of
points. See also [`approximation_name`](@ref).
"""
function approximation_short_name end

"""
    approximate_meancov(method, g, m::Real, v::Real) -> (mean, variance)
    approximate_meancov(method, g, m::AbstractVector, P::AbstractMatrix) -> (mean, covariance)

The mean and variance (or covariance) of the density proportional to `g(x) N(x | m, v)`,
computed by the cubature `method` over the points of `N(m, v)`. This is the projection of a
normal reweighted by a likelihood onto a normal, the core of an expectation-propagation update.

# Arguments

- `method`: a cubature, such as [`GaussHermiteCubature`](@ref)`(p)`; anything with
  [`getpoints`](@ref) and [`getweights`](@ref);
- `g`: a non-negative function of a point, a number for a scalar normal and a vector otherwise.
  It need not be normalised, but must not vanish at every point;
- `m`, `v` or `P`: the mean and variance, or mean vector and covariance matrix, of the normal.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> m, v = approximate_meancov(ghcubature(21), x -> exp(-x^2 / 2), 0.0, 1.0);  # N(0, 1) × N(0, 1)

julia> isapprox(m, 0.0; atol = 1e-12) && v ≈ 0.5
true
```
"""
function approximate_meancov end

function approximate_meancov(method::AbstractApproximationMethod, g::G, m::T, v::T) where {G, T <: Real}
    weights = getweights(method, m, v)
    points = getpoints(method, m, v)

    cs = Vector{eltype(m)}(undef, length(weights))
    norm = 0.0
    mean = 0.0
    for (index, (weight, point)) in enumerate(zip(weights, points))
        cv = weight * g(point)
        mean += point * cv
        norm += cv
        @inbounds cs[index] = cv
    end
    mean /= norm

    var = 0.0
    for (point, c) in zip(points, cs)
        var += c * (point - mean)^2
    end
    var /= norm

    return mean, var
end

function approximate_meancov(method::AbstractApproximationMethod, g::G, m::AbstractVector{T}, P::AbstractMatrix{T}) where {G, T <: Real}
    ndims = length(m)
    weights = getweights(method, m, P)
    points = getpoints(method, m, P)

    cs = similar(m, eltype(m), length(weights))
    norm = zero(T)
    mean = zeros(T, ndims)
    for (index, (weight, point)) in enumerate(zip(weights, points))
        cv = weight * g(point)
        # The point is one buffer the next iteration rewrites, so it may be scaled in place.
        broadcast!(*, point, point, cv)
        broadcast!(+, mean, mean, point)
        norm += cv
        @inbounds cs[index] = cv
    end
    broadcast!(/, mean, mean, norm)

    cov = zeros(T, ndims, ndims)
    foreach(zip(points, cs)) do (point, c)
        broadcast!(-, point, point, mean)
        mul!(cov, point, reshape(point, (1, ndims)), c, 1.0)
    end
    broadcast!(/, cov, cov, norm)

    return mean, cov
end

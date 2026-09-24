"""
    AbstractApproximationMethod

Supertype of the moment-propagation methods, such as [`Unscented`](@ref).
"""
abstract type AbstractApproximationMethod end

"""
    approximation_name(method)

The method's full name, for display.
"""
function approximation_name end

"""
    approximation_short_name(method)

The method's abbreviated name, for display.
"""
function approximation_short_name end

"""
    approximate_meancov(method, g, m, v)
    approximate_meancov(method, g, m::AbstractVector, P::AbstractMatrix)

The mean and variance (or covariance) of the distribution proportional to `g(x) N(x | m, v)`,
by the cubature `method`, such as a [`GaussHermiteCubature`](@ref). `g` need not be normalised.
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

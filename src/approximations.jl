export AbstractApproximationMethod
export approximation_name, approximation_short_name

abstract type AbstractApproximationMethod end

function approximation_name end
function approximation_short_name end

function approximate_meancov(method::AbstractApproximationMethod, g::Function, distribution)
    return approximate_meancov(method, g, mean(distribution), cov(distribution))
end

function approximate_meancov(method::AbstractApproximationMethod, g::Function, m::T, v::T) where { T <: Real }
    weights = getweights(method, m, v)
    points  = getpoints(method, m, v)

    cs   = Vector{eltype(m)}(undef, length(weights))
    norm = 0.0
    mean = 0.0

    for (index, (weight, point)) in enumerate(zip(weights, points))
        gv = g(point)
        cv = weight * gv

        mean += point * cv
        norm += cv

        @inbounds cs[index] = cv
    end

    mean /= norm

    var = 0.0
    for (index, (point, c)) in enumerate(zip(points, cs))
        point -= mean
        var += c * point ^ 2
    end

    var /= norm

    return mean, var
end

function approximate_meancov(method::AbstractApproximationMethod, g::Function, m::AbstractVector{T}, P::AbstractMatrix{T}) where { T <: Real }
    ndims = length(m)

    weights = getweights(method, m, P)
    points  = getpoints(method, m, P)

    cs = similar(m, eltype(m), length(weights))
    norm = zero(T)
    mean = zeros(T, ndims)

    for (index, (weight, point)) in enumerate(zip(weights, points))
        gv = g(point)
        cv = weight * gv

        # mean = mean + point * weight * g(point)
        broadcast!(*, point, point, cv)  # point *= cv
        broadcast!(+, mean, mean, point) # mean += point
        norm += cv

        @inbounds cs[index] = cv
    end

    broadcast!(/, mean, mean, norm)

    cov = zeros(T, ndims, ndims)
    foreach(enumerate(zip(points, cs))) do (index, (point, c))
        broadcast!(-, point, point, mean)                # point -= mean
        mul!(cov, point, reshape(point, (1, ndims)), c, 1.0) # cov = cov + c * (point)⋅(point)' where c = weight * g(point)
    end

    broadcast!(/, cov, cov, norm)

    return mean, cov
end

function approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, distribution)
    return approximate_kernel_expectation(method, g, mean(distribution), cov(distribution))
end

function approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, m::AbstractVector{T}, P::AbstractMatrix{T}) where { T <: Real }
    ndims = length(m)

    weights = getweights(method, m, P)
    points  = getpoints(method, m, P)

    gbar = zeros(ndims, ndims)
    foreach(zip(weights, points)) do (weight, point)
        axpy!(weight, g(point), gbar) # gbar = gbar + weight * g(point)
    end

    return gbar
end
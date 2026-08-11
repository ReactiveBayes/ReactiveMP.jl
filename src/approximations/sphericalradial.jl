export srcubature

struct SphericalRadialCubature <: AbstractApproximationMethod end

approximation_name(::SphericalRadialCubature)       = "SphericalRadial"
approximation_short_name(::SphericalRadialCubature) = "SR"

function srcubature()
    return SphericalRadialCubature()
end

function getweights(
    ::SphericalRadialCubature,
    mean::AbstractVector{T},
    covariance::AbstractMatrix{T},
) where {T <: Real}
    d = length(mean)
    return Base.Generator(1:(2d + 1)) do i
        return i === (2d + 1) ? 1.0 / (d + 1) : 1.0 / (2.0(d + 1))
    end
end

# getpoints(::SphericalRadialCubature, mean::AbstractVector, covariance::AbstractMatrix)
#
# Return a lazy generator over the `2d + 1` spherical-radial cubature points.
#
# WARNING: the generator yields the same buffer every iteration. Two preallocated vectors are
# reused across iterations -- `tmpbuffer` for the unit sigma point and `tbuffer` for the
# transformed one -- so the generator yields the same object each time with new contents. Two
# consequences:
#
#   * Do not materialize the result. `collect(getpoints(...))` returns `2d + 1` references to one
#     buffer, i.e. that many copies of the last point. The last point here is the centre, so a
#     collected result is the mean repeated -- silently wrong cubature with no error. Use
#     `map(copy, getpoints(...))` if you need independent points.
#   * Consumers may destroy the contents. `approximate_meancov` deliberately mutates the yielded
#     point in place (`broadcast!(*, point, point, cv)`) rather than allocating; that is safe only
#     because the next iteration rewrites the buffer before use.
function getpoints(
    ::SphericalRadialCubature,
    mean::AbstractVector{T},
    covariance::AbstractMatrix{T},
) where {T <: Real}
    d = length(mean)
    L = cholsqrt(covariance)

    tmpbuffer = zeros(d)
    sigma_points = Base.Generator(1:(2d + 1)) do i
        if i === (2d + 1)
            fill!(tmpbuffer, 0.0)
        else
            tmpbuffer[rem((i - 1), d) + 1] = sqrt(d + 1) * (-1)^(div(i - 1, d))
            if i !== 1
                tmpbuffer[rem((i - 2), d) + 1] = 0.0
            end
        end
        return tmpbuffer
    end

    tbuffer = similar(mean)
    return Base.Generator(sigma_points) do point
        copyto!(tbuffer, mean)
        return mul!(tbuffer, L, point, 1.0, 1.0) # point = m + 1.0 * L * point
    end
end

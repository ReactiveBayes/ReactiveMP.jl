export srcubature

struct SphericalRadialCubature <: AbstractApproximationMethod end

approximation_name(::SphericalRadialCubature)       = "SphericalRadial"
approximation_short_name(::SphericalRadialCubature) = "SR"

function srcubature()
    return SphericalRadialCubature()
end

function getweights(::SphericalRadialCubature, mean::AbstractVector{T}, covariance::AbstractPDMat{T}) where { T <: Real }
    d = length(mean)
    return Base.Generator(1:2d + 1) do i
        return i === (2d + 1) ? 1.0 / (d + 1) : 1.0 / (2.0(d + 1))
    end
end

function getpoints(::SphericalRadialCubature, mean::AbstractVector{T}, covariance::AbstractPDMat{T}) where { T <: Real }
    d = length(mean)
    L = sqrt(Hermitian(covariance))

    tmpbuffer = zeros(d)
    sigma_points = Base.Generator(1:2d + 1) do i
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
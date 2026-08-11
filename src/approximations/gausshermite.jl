export ghcubature, GaussHermiteCubature

import FastGaussQuadrature: gausshermite
import LinearAlgebra: mul!, axpy!

using Distributions

const product  = Iterators.product
const repeated = Iterators.repeated
const sqrtPI1  = sqrt(pi)

struct GaussHermiteCubature{PI, WI} <: AbstractApproximationMethod
    piter::PI
    witer::WI
end

GaussHermiteCubature(p::Int) = ghcubature(p)

approximation_name(approx::GaussHermiteCubature)       = "GaussHermite($(approx.p))"
approximation_short_name(approx::GaussHermiteCubature) = "GH$(approx.p)"

function ghcubature(p::Int)
    points, weights = gausshermite(p)
    return GaussHermiteCubature(points, weights)
end

function getweights(
    gh::GaussHermiteCubature, mean::T, variance::T
) where {T <: Real}
    return Base.Generator(gh.witer) do weight
        return weight / sqrtPI1
    end
end

function getweights(
    gh::GaussHermiteCubature,
    mean::AbstractVector{T},
    covariance::AbstractMatrix{T},
) where {T <: Real}
    sqrtpi = (pi^(length(mean) / 2))
    return Base.Generator(
        product(repeated(gh.witer, length(mean))...)
    ) do pweight
        return prod(pweight) / sqrtpi
    end
end

function getpoints(
    gh::GaussHermiteCubature, mean::T, variance::T
) where {T <: Real}
    sqrt2V = sqrt(2 * variance)
    return Base.Generator(gh.piter) do point
        return mean + sqrt2V * point
    end
end

# getpoints(cubature::GaussHermiteCubature, mean::AbstractVector, covariance::AbstractMatrix)
#
# Return a lazy generator over the multivariate Gauss-Hermite cubature points.
#
# WARNING: the generator yields the same buffer every iteration. For performance, every point is
# written into a single preallocated vector that is reused across iterations, so the generator
# yields the same object each time with new contents. Two consequences:
#
#   * Do not materialize the result. `collect(getpoints(...))` returns `n` references to that one
#     buffer, i.e. `n` copies of the last point — silently wrong cubature with no error. Use
#     `map(copy, getpoints(...))` if you need independent points.
#   * Consumers may destroy the contents. `approximate_meancov` deliberately mutates the yielded
#     point in place (`broadcast!(*, point, point, cv)`) rather than allocating; that is safe only
#     because the next iteration rewrites the buffer from `mean` before use.
#
# The univariate method above does not have this property: it yields freshly computed scalars.
function getpoints(
    cubature::GaussHermiteCubature,
    mean::AbstractVector{T},
    covariance::AbstractMatrix{T},
) where {T <: Real}
    sqrtP = cholsqrt(covariance)
    sqrt2 = sqrt(2)

    tbuffer = similar(mean)
    pbuffer = similar(mean)
    return Base.Generator(
        product(repeated(cubature.piter, length(mean))...)
    ) do ptuple
        copyto!(pbuffer, ptuple)
        copyto!(tbuffer, mean)
        return mul!(tbuffer, sqrtP, pbuffer, sqrt2, 1.0) # point = m + sqrt2 * sqrtP * p
    end
end

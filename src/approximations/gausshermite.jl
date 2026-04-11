export ghcubature, GaussHermiteCubature

import FastGaussQuadrature: gausshermite
import LinearAlgebra: mul!, axpy!

using Distributions

const product  = Iterators.product
const repeated = Iterators.repeated
const sqrtPI1  = sqrt(pi)

"""
    GaussHermiteCubature{PI, WI} <: AbstractApproximationMethod

Gauss-Hermite cubature approximation method. Computes expectations of the form
`∫ g(x) N(x; μ, σ²) dx` using a fixed set of quadrature points and weights
optimised for Gaussian measures. The approximation is exact for polynomials up
to degree `2p - 1`, where `p` is the number of points.

Use [`ghcubature`](@ref) to construct an instance with a given number of points.
"""
struct GaussHermiteCubature{PI, WI} <: AbstractApproximationMethod
    piter::PI
    witer::WI
end

GaussHermiteCubature(p::Int) = ghcubature(p)

approximation_name(approx::GaussHermiteCubature)       = "GaussHermite($(approx.p))"
approximation_short_name(approx::GaussHermiteCubature) = "GH$(approx.p)"

"""
    ghcubature(p::Int) -> GaussHermiteCubature

Construct a [`GaussHermiteCubature`](@ref) rule with `p` quadrature points and
the corresponding weights for Gauss-Hermite integration.
"""
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

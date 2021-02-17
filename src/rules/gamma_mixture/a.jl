import SpecialFunctions: digamma


struct GammaAConjugate{T <: Real}
    π :: T
    γ :: T # π * β
    r :: T # log(Z)
    a :: Int
end


function computeZ1(approximation::GaussLaguerreQuadrature, f::Function)
    g = (x) -> f(x)

    points  = getpoints(approximation, nothing, nothing)
    weights = getweights(approximation, nothing, nothing)

    Z = mapreduce(+, zip(points, weights)) do (p, w)
        return w * g(p)
    end

    return Z
end

function prod(::ProdPreserveParametrisation, left::GammaAConjugate, right::GammaAConjugate)
    return GammaAConjugate(left.π + right.π, left.γ + right.γ, left.r + right.r, max(left.a, right.a))
end

function prod(::ProdPreserveParametrisation, left::GammaAConjugate, right::GammaDistributionsFamily)
    return prod(ProdPreserveParametrisation(), right, left)
end

function prod(::ProdPreserveParametrisation, left::GammaDistributionsFamily, right::GammaAConjugate)
    η = rate(left)

    f = (x) -> begin return exp(right.γ * x - right.π * loggamma(x)) * exp((shape(left) - 1) * log(x)) end
    gl = glquadrature(right.a)

    # @show right

    # @show shape(left), rate(left)

    Z = computeZ1(gl, x -> f(x / η) / η)
    m = computeZ1(gl, x -> (x / η) * f(x / η) / Z / η)
    v = computeZ1(gl, x -> (x / η  - m) ^ 2 * f(x / η) / Z / η)

    # @show left, right
    # @show η, Z
    # @show m, v

    # @show Z
    #@show m, v
    # v = clamp(computeZ1(gl, x -> (x - m) ^ 2 * f(x) / Z), var(left), huge)
    # s = computeZ1(gl, x -> (x - m) ^ 3 * f(x) / Z)

    # zz1 = computeZ1(glquadrature(150), (x) -> exp(right.γ * x - right.π * loggamma(x) - right.r))
    # @show zz1

    # @show left
    # @show right

    a = m ^ 2 / v
    b = m / v

    # @show left, right
    # @show η, a, b, m, v, Z

    # @show right.π, right.γ, right.r, shape(left), rate(left)
    # @show Z, m, v, a, b, right.r

    return GammaShapeRate(a, b)
end


@rule GammaMixture((:a, k), Marginalisation) (q_out::Any, q_switch::Any, q_a::NTuple{N1, GammaDistributionsFamily }, q_b::NTuple{N2, GammaDistributionsFamily}) where { N1, N2 } = begin
    π = probvec(q_switch)[k]
    β = logmean(q_out) + logmean(q_b[k])
    γ = π * β

    # @assert (N1, N2) === (1, 2)

    # Z = computeZ1(glquadrature(120), x -> exp((γ + 1) * x - π * loggamma(x)))
    # r = log(Z)

    # zz2 = computeZ1(glquadrature(150), (x) -> exp(γ * x - π * loggamma(x) - r))
    # @show zz2

    return GammaAConjugate(π, γ, zero(π), 120)
    # TODO: Needs further discussion, doublecheck
    # â = mean(q_out) * mean(q_b[k])
    # â = mean(getrecent(ReactiveMP.getmarginal(connectedvar(__node.as[k]))))
    # Ψ = logmean(q_out) + logmean(q_b[k]) - digamma(â)
    # return GammaShapeRate(1, probvec(q_switch)[k]*Ψ)
end

# import SpecialFunctions: digamma

# using DomainIntegrals
# using DomainSets

# struct GammaAConjugate{T <: Real}
#     π :: T
#     γ :: T # π * β
#     r :: T # log(Z)
#     a :: Int
# end


# function computeZ1(approximation::GaussLaguerreQuadrature, f::Function)
#     g = (x) -> exp(x)*f(x)
    
#     points  = getpoints(approximation, nothing, nothing)
#     weights = getweights(approximation, nothing, nothing)
    
#     Z = mapreduce(+, zip(points, weights)) do (p, w)
#         return w * g(p)
#     end
    
#     return Z
# end

# function prod(::ProdPreserveParametrisation, left::GammaAConjugate, right::GammaAConjugate)
#     return GammaAConjugate(left.π + right.π, left.γ + right.γ, left.r + right.r, max(left.a, right.a))
# end

# function prod(::ProdPreserveParametrisation, left::GammaAConjugate, right::GammaDistributionsFamily)
#     return prod(ProdPreserveParametrisation(), right, left)
# end

# function prod(::ProdPreserveParametrisation, left::GammaDistributionsFamily, right::GammaAConjugate)
#     f = (x) -> exp(right.γ * x - right.π * loggamma(x) - right.r) * exp((shape(left) - 1) * log(x) - rate(left) * x)
#     gl = glquadrature(right.a)

#     Z = integral(f, HalfLine())
#     m = integral(x -> x * f(x) / Z, HalfLine())
#     v = integral(x -> (x - m)^2 * f(x) / Z, HalfLine())
#     # b̂ = abs(rate(left) - right.γ)
#     # â = m * b̂

#     # Z = computeZ1(gl, f)
#     # m = computeZ1(gl, x -> x * f(x) / Z)
#     # v = computeZ1(gl, x -> x ^ 2 * f(x) / Z) - m^2
#     # s = computeZ1(gl, x -> (x - m) ^ 3 * f(x) / Z)

#     a = m ^ 2 / v
#     # a = (2 / s) ^ 2
#     b = m / v

#     # @show right.π, right.γ, right.r, shape(left), rate(left)
#     # @show a, b

#     # return GammaShapeRate(â, b̂)
#     return GammaShapeRate(a, b)
# end


# @rule GammaMixture((:a, k), Marginalisation) (q_out::Any, q_switch::Any, q_a::NTuple{N1, GammaDistributionsFamily }, q_b::NTuple{N2, GammaDistributionsFamily}) where { N1, N2 } = begin
#     π = probvec(q_switch)[k]
#     β = logmean(q_out) + logmean(q_b[k])
#     γ = π * β
#     Z = computeZ1(glquadrature(150), (x) -> exp(γ * x - π * loggamma(x)))
#     r = log(Z)
#     return GammaAConjugate(π, γ, r, 150)
#     # TODO: Needs further discussion, doublecheck
#     # â = mean(q_out) * mean(q_b[k])
#     # â = mean(getrecent(ReactiveMP.getmarginal(connectedvar(__node.as[k]))))
#     # Ψ = logmean(q_out) + logmean(q_b[k]) - digamma(â)
#     # return GammaShapeRate(1, probvec(q_switch)[k]*Ψ)
# end
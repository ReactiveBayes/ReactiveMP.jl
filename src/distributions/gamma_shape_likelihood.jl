
import SpecialFunctions: loggamma

"""
    ν(x) ∝ exp(p*β*x - π*logГ(x)) ≡ exp(γ*x - p*logГ(x))
"""
struct GammaShapeLikelihood{T <: Real, A}
    p :: T
    γ :: T # p * β
    
    approximation :: A
end

function approximate_prod_expectations(approximation::GaussLaguerreQuadrature, left::GammaDistributionsFamily, right::GammaShapeLikelihood)
    b = rate(left)

    f = let γ = right.γ, p = right.p, a = shape(left)
        x -> exp(γ * x - p * loggamma(x) + (a - 1) * log(x))
    end

    Cf = let f = f, b = b 
        x -> f(x / b) / b
    end 

    C = approximate(approximation, Cf)

    mf = let f = f, b = b, C = C
        x -> (x / b) * f(x / b) / (C * b)
    end

    m = approximate(approximation, mf)

    vf = let f = f, b = b, C = C, m = m
        x -> (x / b - m) ^ 2 * f(x / b) / (C * b)
    end

    v = approximate(approximation, vf)

    return C, m, v
end

function prod(::ProdPreserveParametrisation, left::GammaShapeLikelihood, right::GammaShapeLikelihood)
    @assert left.approximation == right.approximation "Different approximation types for $(left) and $(right) messages"
    return GammaShapeLikelihood(left.p + right.p, left.γ + right.γ, left.approximation)
end

function prod(::ProdPreserveParametrisation, left::GammaShapeLikelihood, right::GammaDistributionsFamily)
    return prod(ProdPreserveParametrisation(), right, left)
end

function prod(::ProdPreserveParametrisation, left::GammaDistributionsFamily, right::GammaShapeLikelihood)
    C, m , v = approximate_prod_expectations(right.approximation, left, right)

    a = m ^ 2 / v
    b = m / v

    return GammaShapeRate(a, b)
end
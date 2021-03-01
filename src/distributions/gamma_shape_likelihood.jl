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

    """
    q(x)    ∝ v(x)*v(x)
            ∝ exp(γ*x - p*ln(Г(x))) * exp((a-1)*ln(x) - b*x)
            = exp(-x) * exp((γ-b+1)*x + (a-1)*ln(x) - p*ln(Г(x)))
    """
    f = let p = right.p, a = shape(left), γ = right.γ, b = b
        x -> exp((γ - b + 1) * x - p * loggamma(x) + (a - 1) * log(x))
    end

    logf = let p = right.p, a = shape(left), γ = right.γ, b = b
        x -> (γ - b + 1) * x - p * loggamma(x) + (a - 1) * log(x)
    end

    # calculate log-normalization constant
    logC = log_approximate(approximation, logf)

    # mean function without explicitly calculating the normalization constant
    mf = let logf = logf, logC = logC
        x -> x * exp(logf(x) - logC)
    end

    # calculate mean
    m = approximate(approximation, mf)

    # variance function without explicitly calculating the normalization constant
    vf = let logf = logf, logC = logC, m = m
        x -> (x - m) ^ 2 * exp(logf(x) - logC)
    end

    # calculate variance
    v = approximate(approximation, vf)

    return logC, m, v
end

function prod(::ProdPreserveParametrisation, left::GammaShapeLikelihood, right::GammaShapeLikelihood)
    @assert left.approximation == right.approximation "Different approximation types for $(left) and $(right) messages"
    return GammaShapeLikelihood(left.p + right.p, left.γ + right.γ, left.approximation)
end

function prod(::ProdPreserveParametrisation, left::GammaShapeLikelihood, right::GammaDistributionsFamily)
    return prod(ProdPreserveParametrisation(), right, left)
end

function prod(::ProdPreserveParametrisation, left::GammaDistributionsFamily, right::GammaShapeLikelihood)
    _, m, v = approximate_prod_expectations(right.approximation, left, right)

    a = m ^ 2 / v
    b = m / v

    return GammaShapeRate(a, b)
end
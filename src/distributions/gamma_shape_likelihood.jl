import SpecialFunctions: loggamma

using FastGaussQuadrature: gausslaguerre
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
    This function calculates the log of the Gauss-laguerre integral by making use of the log of the integrable function.
    ln ( ∫ exp(-x)f(x) dx ) 
    ≈ ln ( ∑ wi * f(xi) ) 
    = ln ( ∑ exp( ln(wi) + logf(xi) ) )
    = ln ( ∑ exp( yi ) )
    = max(yi) + ln ( ∑ exp( yi - max(yi) ) )
    where we make use of the numerically stable log-sum-exp trick: https://en.wikipedia.org/wiki/LogSumExp
    """
    function loggausslaguerre(logCf, p)

        # calculate weights and points
        x, w = gausslaguerre(p)

        # calculate the ln(wi) + logf(xi) terms
        logresult = Array{Float64,1}(undef, p)
        for i = 1:p
            logresult[i] = log(w[i]) + logCf(x[i])
        end

        # log-sum-exp trick, calculate maximum
        max_logresult = maximum(logresult)

        # return log sum exp
        return max_logresult + log(sum(exp.(logresult .- max_logresult)))
    end

    """
    q(x)    ∝ v(x)*v(x)
            ∝ exp(γ*x - p*ln(Г(x))) * exp((a-1)*ln(x) - b*x)
            = exp(-x) * exp((γ-b+1)*x + (a-1)*ln(x) - p*ln(Г(x)))
    """
    f = let p = right.p, a = shape(left), γ=right.γ, b=b
        x -> exp((γ-b+1)*x - p * loggamma(x) + (a - 1) * log(x))
    end
    logf = let p = right.p, a = shape(left), γ=right.γ, b=b
        x -> (γ-b+1)*x - p * loggamma(x) + (a - 1) * log(x)
    end

    # calculate log-normalization constant
    p = length(approximation.rule.x)
    logC =  loggausslaguerre(logf, p)

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
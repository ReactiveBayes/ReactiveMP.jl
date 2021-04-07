import SpecialFunctions: loggamma
using Distributions
using Optim

"""
    ν(x) ∝ exp(p*β*x - p*logГ(x)) ≡ exp(γ*x - p*logГ(x))
"""
struct GammaShapeLikelihood{T <: Real, A}
    p :: T
    γ :: T # p * β

    approximation :: A
end

Base.show(io::IO, distribution::GammaShapeLikelihood{T}) where T = print(io, "GammaShapeLikelihood{$T}(π = $(distribution.p), γ = $(distribution.γ))")

Distributions.logpdf(distribution::GammaShapeLikelihood, x) = distribution.γ * x - distribution.p * loggamma(x)

# function approximate_prod_expectations(approximation::GaussLaguerreQuadrature, left::GammaDistributionsFamily, right::GammaShapeLikelihood)
#     b = rate(left)

#     """
#     q(x)    ∝ v(x)*v(x)
#             ∝ exp(γ*x - p*ln(Г(x))) * exp((a-1)*ln(x) - b*x)
#             = exp(-x) * exp((γ-b+1)*x + (a-1)*ln(x) - p*ln(Г(x)))
#     """
#     f = let p = right.p, a = shape(left), γ = right.γ, b = b
#         x -> exp((γ - b + 1) * x - p * loggamma(x) + (a - 1) * log(x))
#     end

#     logf = let p = right.p, a = shape(left), γ = right.γ, b = b
#         x -> (γ - b + 1) * x - p * loggamma(x) + (a - 1) * log(x)
#     end

#     # calculate log-normalization constant
#     logC = log_approximate(approximation, logf)

#     # mean function without explicitly calculating the normalization constant
#     mf = let logf = logf, logC = logC
#         x -> x * exp(logf(x) - logC)
#     end

#     # calculate mean
#     m = approximate(approximation, mf)

#     # variance function without explicitly calculating the normalization constant
#     vf = let logf = logf, logC = logC, m = m
#         x -> (x - m) ^ 2 * exp(logf(x) - logC)
#     end

#     # calculate variance
#     v = approximate(approximation, vf)

#     return logC, m, v
# end
function approximate_prod_expectations(left::GammaDistributionsFamily, right::GammaShapeLikelihood)
    b = scale(left)
    a = shape(left)

    f = let p = right.p, γ = right.γ
        x -> exp(γ * x - p * loggamma(x))
    end

    sampling_dist = Distributions.Gamma(a,b)
    # @show sampling_dist
    samples = rand(sampling_dist, 10000)
    transformed_samples = f.(samples) 
    # @show transformed_samples
    normalization = sum(transformed_samples)
    weights = transformed_samples ./ sum(transformed_samples)
    
    m = sum(weights .* samples)
    v = sum(weights .* (samples .- m) .^ 2)
    # @show m,v
    return m, v
end



# Preserve Gamma Distribution

function prod(::ProdPreserveParametrisation, left::GammaShapeLikelihood, right::GammaShapeLikelihood)
    @assert left.approximation == right.approximation "Different approximation types for $(left) and $(right) messages"
    return GammaShapeLikelihood(left.p + right.p, left.γ + right.γ, left.approximation)
end

function prod(::ProdPreserveParametrisation, left::GammaShapeLikelihood, right::GammaDistributionsFamily)
    return prod(ProdPreserveParametrisation(), right, left)
end

function prod(::ProdPreserveParametrisation, left::GammaDistributionsFamily, right::GammaShapeLikelihood)
    m, v = approximate_prod_expectations(left, right)

    a = m ^ 2 / v
    b = m / v

    return GammaShapeRate(a, b)
end

# Expectation maximisation

function prod(::ProdExpectationMaximisation, left::GammaShapeLikelihood, right::GammaShapeLikelihood)
    @assert left.approximation == right.approximation "Different approximation types for $(left) and $(right) messages"
    return GammaShapeLikelihood(left.p + right.p, left.γ + right.γ, left.approximation)
end

function prod(::ProdExpectationMaximisation, left::GammaShapeLikelihood, right::GammaDistributionsFamily)
    return prod(ProdPreserveParametrisation(), right, left)
end

function prod(::ProdExpectationMaximisation, left::GammaDistributionsFamily, right::GammaShapeLikelihood)

    a, b = shape(left), rate(left)
    γ, p = right.γ, right.p
    f(x) = (a-1)*log(x[1]) - b*x[1] + γ*x[1] - p*loggamma(x[1]) - loggamma(a) + a*log(b)
    x_0 = [mean(left)]
    res = optimize(x -> -f(x), [ 0.0 ], [ Inf ], x_0, Fminbox(GradientDescent()))
    # res = optimize(x -> -f(x), x_0, Newton())
    â = Optim.minimizer(res)[1]
    # @show â

    return PointMass(â)
end

import SpecialFunctions: loggamma
using Distributions
using Optim

"""
    ν(x) ∝ exp(p*β*x - p*logГ(x)) ≡ exp(γ*x - p*logГ(x))
"""
struct GammaShapeLikelihood{T <: Real} <: ContinuousUnivariateDistribution
    p::T
    γ::T # p * β
end

Distributions.@distr_support GammaShapeLikelihood 0 Inf

Distributions.support(dist::GammaShapeLikelihood) = Distributions.RealInterval(minimum(dist), maximum(dist))

Base.show(io::IO, distribution::GammaShapeLikelihood{T}) where {T} =
    print(io, "GammaShapeLikelihood{$T}(π = $(distribution.p), γ = $(distribution.γ))")

Distributions.logpdf(distribution::GammaShapeLikelihood, x::Real) = distribution.γ * x - distribution.p * loggamma(x)

prod_analytical_rule(::Type{<:GammaShapeLikelihood}, ::Type{<:GammaShapeLikelihood}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::GammaShapeLikelihood, right::GammaShapeLikelihood)
    return GammaShapeLikelihood(left.p + right.p, left.γ + right.γ)
end

# TODO over form constraints
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

#     return m, v
# end

# function approximate_prod_expectations(approximation::ImportanceSamplingApproximation, left::GammaDistributionsFamily, right::GammaShapeLikelihood)

#     f = let p = right.p, γ = right.γ
#         x -> exp(γ * x - p * loggamma(x))
#     end

#     m, v = approximate_meancov(approximation, f, GammaShapeScale(shape(left), scale(left)))

#     return m, v
# end

# function prod(::ProdAnalytical, left::GammaShapeLikelihood, right::GammaDistributionsFamily)
#     return prod(ProdAnalytical(), right, left)
# end

# function prod(::ProdAnalytical, left::GammaDistributionsFamily, right::GammaShapeLikelihood)
#     m, v = approximate_prod_expectations(right.approximation, left, right)

#     a = m ^ 2 / v
#     b = m / v

#     return GammaShapeRate(a, b)
# end

# Expectation maximisation
# TODO: same do over form constraints

# function prod(::ProdExpectationMaximisation, left::GammaShapeLikelihood, right::GammaShapeLikelihood)
#     @assert left.approximation == right.approximation "Different approximation types for $(left) and $(right) messages"
#     return GammaShapeLikelihood(left.p + right.p, left.γ + right.γ, left.approximation)
# end

# function prod(::ProdExpectationMaximisation, left::GammaShapeLikelihood, right::GammaDistributionsFamily)
#     return prod(ProdExpectationMaximisation(), right, left)
# end

# function prod(::ProdExpectationMaximisation, left::GammaDistributionsFamily, right::GammaShapeLikelihood)

#     a, b = shape(left), rate(left)
#     γ, p = right.γ, right.p

#     f(x) = (a-1)*log(x[1]) - b*x[1] + γ*x[1] - p*loggamma(x[1]) - loggamma(a) + a*log(b)

#     x_0 = [ mean(left) ]
#     res = optimize(x -> -f(x), [ 0.0 ], [ Inf ], x_0, Fminbox(GradientDescent()))

#     â = Optim.minimizer(res)[1]

#     return PointMass(â)
# end

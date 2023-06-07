import SpecialFunctions: besselk

@rule typeof(*)(:out, Marginalisation) (m_A::PointMass, m_in::PointMass, meta::Union{<:AbstractCorrection, Nothing}) = PointMass(mean(m_A) * mean(m_in))

@rule typeof(*)(:out, Marginalisation) (m_A::PointMass{<:Real}, m_in::GammaDistributionsFamily, meta::Union{<:AbstractCorrection, Nothing}) = begin
    return GammaShapeRate(shape(m_in), rate(m_in) / mean(m_A))
end

@rule typeof(*)(:out, Marginalisation) (m_A::GammaDistributionsFamily, m_in::PointMass{<:Real}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    return @call_rule typeof(*)(:out, Marginalisation) (m_A = m_in, m_in = m_A, meta = meta, addons = getaddons()) # symmetric rule
end

@rule typeof(*)(:out, Marginalisation) (m_A::PointMass{<:AbstractMatrix}, m_in::F, meta::Union{<:AbstractCorrection, Nothing}) where {F <: NormalDistributionsFamily} = begin
    @logscale 0
    A = mean(m_A)
    μ_in, Σ_in = mean_cov(m_in)
    return convert(promote_variate_type(F, NormalMeanVariance), A * μ_in, A * Σ_in * A')
end

@rule typeof(*)(:out, Marginalisation) (m_A::F, m_in::PointMass{<:AbstractMatrix}, meta::Union{<:AbstractCorrection, Nothing}) where {F <: NormalDistributionsFamily} = begin
    return @call_rule typeof(*)(:out, Marginalisation) (m_A = m_in, m_in = m_A, meta = meta, addons = getaddons()) # symmetric rule
end

#------------------------
# AbstractVector * UnivariateNormalDistributions
#------------------------
# We consider the following updates as a special case of the MatrixVariate * Multivariate updates.
# Namely, Ax = y, where A ∈ R^{nx1}, x ∈ R^1, and y ∈ R^n. In this case, the matrix A
# can be represented by a n-dimensional vector, and x by a scalar. Before computation,
# quantities are converted to their proper dimensions (see situational sketch below).
#
#     | a ~ AbstractVector -> R^{nx1}
#     v  out ~ Multivariate -> R^n
# -->[x]-->
# in1 ~ Univariate -> R^1
@rule typeof(*)(:out, Marginalisation) (m_A::PointMass{<:AbstractVector}, m_in::UnivariateNormalDistributionsFamily, meta::Union{<:AbstractCorrection, Nothing}) = begin
    @logscale 0
    a = mean(m_A)

    μ_in, v_in = mean_var(m_in)

    # TODO: check, do we need correction! here? (ForneyLab does not have any correction in this case)
    # TODO: Σ in this rule is guaranteed to be ill-defined, has rank equal to one and has determinant equal to zero
    μ = μ_in * a
    Σ = mul_inplace!(v_in, a * a')

    return MvNormalMeanCovariance(μ, Σ)
end

@rule typeof(*)(:out, Marginalisation) (m_A::UnivariateNormalDistributionsFamily, m_in::PointMass{<:AbstractVector}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    return @call_rule typeof(*)(:out, Marginalisation) (m_A = m_in, m_in = m_A, meta = meta, addons = getaddons()) # symmetric rule
end

#------------------------
# Real * UnivariateNormalDistributions
#------------------------
@rule typeof(*)(:out, Marginalisation) (m_A::PointMass{<:Real}, m_in::UnivariateNormalDistributionsFamily, meta::Union{<:AbstractCorrection, Nothing}) = begin
    @logscale 0
    a = mean(m_A)
    μ_in, v_in = mean_var(m_in)
    return NormalMeanVariance(a * μ_in, a^2 * v_in)
end

@rule typeof(*)(:out, Marginalisation) (m_A::UnivariateNormalDistributionsFamily, m_in::PointMass{<:Real}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    return @call_rule typeof(*)(:out, Marginalisation) (m_A = m_in, m_in = m_A, meta = meta, addons = getaddons()) # symmetric rule
end

#-----------------------
# Univariate Normal * Univariate Normal 
#----------------------
@rule typeof(*)(:out, Marginalisation) (m_A::UnivariateGaussianDistributionsFamily, m_in::UnivariateGaussianDistributionsFamily, meta::Union{<:AbstractCorrection, Nothing}) = begin
    μ_A, var_A = mean_var(m_A)
    μ_in, var_in = mean_var(m_in)

    return ContinuousUnivariateLogPdf(besselmod(μ_in, var_in, μ_A, var_A, 0.0))
end

@rule typeof(*)(:out, Marginalisation) (m_A::UnivariateDistribution, m_in::UnivariateDistribution, meta::Union{<:AbstractCorrection, Nothing}) = begin
    nsamples = 3000
    samples_A = rand(m_A, nsamples)
    p = make_productdist_message(samples_A, m_in)

    return ContinuousUnivariateLogPdf(p)
end

"""
Modified-bessel function of second kind

mx, vx : mean and variance of the random variable x 
my, vy : mean and variance of the random variable y 
rho    : correlation coefficient
"""
function besselmod(mx, vx, my, vy, rho; truncation = 10, jitter = 1e-8)
    logpdf = function (x)
        x += jitter
        term1 = -1 / (2 * (1 - rho^2)) * (mx^2 / vx + my^2 / vy - 2 * rho * (x + mx * my) / sqrt(vx * vy))

        term2 = 0.0
        for n in 0:truncation
            for m in 0:(2 * n)
                term2 +=
                    x^(2 * n - m) * abs(x)^(m - n) * sqrt(vx)^(m - n - 1) / (pi * factorial(2 * n) * (1 - rho^2)^(2 * n + 1 / 2) * sqrt(vy)^(m - n + 1)) *
                    (mx / vx - rho * my / sqrt(vx * vy))^m *
                    binomial(2 * n, m) *
                    (my / vy - rho * mx / sqrt(vx * vy))^(2 * n - m) *
                    besselk(m - n, abs(x) / ((1 - rho^2) * sqrt(vx * vy)))
            end
        end
        return term1 + log(term2)
    end
    return logpdf
end

function make_productdist_message(samples_A, d_in)
    return let samples_A = samples_A, d_in = d_in
        (x) -> begin
            result = mapreduce(+, samples_A) do sampleA
                return 1 / abs(sampleA) * pdf(d_in, x / sampleA)
            end
            return log(result)
        end
    end
end

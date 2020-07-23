export averageEnergy, differentialEntropy

using SpecialFunctions

# (m_mean, v_mean) = unsafeMeanCov(marg_mean)
#     (m_out, v_out) = unsafeMeanCov(marg_out)
#
#     0.5*log(2*pi) -
#     0.5*unsafeLogMean(marg_prec) +
#     0.5*unsafeMean(marg_prec)*(v_out + v_mean + (m_out - m_mean)^2)

## Normal

# Marginals: mean, precision, value
function averageEnergy(::Type{<:NormalMeanPrecision}, marginals::Tuple{Marginal, Marginal, Marginal})
    m_mean, v_mean = mean(marginals[1]), var(marginals[1])
    m_out, v_out = mean(marginals[3]), var(marginals[3])

    result = 0.5 * log(2 * π) -
        0.5 * log(mean(marginals[2])) +
        0.5 * mean(marginals[2]) * (v_out + v_mean + (m_out - m_mean)^2)
    return result
end

## Gamma
# marg_out::ProbabilityDistribution{Univariate}, marg_a::ProbabilityDistribution{Univariate, PointMass}, marg_b::ProbabilityDistribution{Univariate}
function averageEnergy(::Type{<:GammaAB}, marginals::Tuple{Marginal{T}, Marginal{T}, Marginal{<:GammaAB}}) where { T <: Real }
    return labsgamma(mean(marginals[1])) - mean(marginals[1]) * log(mean(marginals[2])) -
    (mean(marginals[1]) - 1.0) * log(mean(marginals[3])) +
    mean(marginals[2]) * mean(marginals[3])
end

##

function differentialEntropy(marginal::Marginal{ <: NormalMeanPrecision })
    return 0.5 * log(var(marginal)) + 0.5 * log(2 * π) + 0.5
end

function labsgamma(x::Number)
    return SpecialFunctions.logabsgamma(x)[1]
end

function differentialEntropy(marginal::Marginal{ <: GammaAB })
    distribution = getdata(marginal)
    return labsgamma(distribution.a) -
    (distribution.a - 1.0) * digamma(distribution.a) -
    log(distribution.b) +
    distribution.a
end

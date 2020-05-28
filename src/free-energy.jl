export averageEnergy, differentialEntropy

using SpecialFunctions

# (m_mean, v_mean) = unsafeMeanCov(marg_mean)
#     (m_out, v_out) = unsafeMeanCov(marg_out)
#
#     0.5*log(2*pi) -
#     0.5*unsafeLogMean(marg_prec) +
#     0.5*unsafeMean(marg_prec)*(v_out + v_mean + (m_out - m_mean)^2)

## Normal

# beliefs: mean, precision, value
function averageEnergy(::Type{<:NormalMeanPrecision}, beliefs::Tuple{Belief, Belief, Belief})
    m_mean, v_mean = mean(beliefs[1]), var(beliefs[1])
    m_out, v_out = mean(beliefs[3]), var(beliefs[3])

    result = 0.5 * log(2 * π) -
        0.5 * log(mean(beliefs[2])) +
        0.5 * mean(beliefs[2]) * (v_out + v_mean + (m_out - m_mean)^2)
    return result
end

## Gamma
# marg_out::ProbabilityDistribution{Univariate}, marg_a::ProbabilityDistribution{Univariate, PointMass}, marg_b::ProbabilityDistribution{Univariate}
function averageEnergy(::Type{<:GammaAB}, beliefs::Tuple{Belief{T1}, Belief{T1}, Belief{<:GammaAB}}) where { T1 <: Real, T2 <: Real }
    return labsgamma(mean(beliefs[1])) - mean(beliefs[1]) * log(mean(beliefs[2])) -
    (mean(beliefs[1]) - 1.0) * log(mean(beliefs[3])) +
    mean(beliefs[2]) * mean(beliefs[3])
end

##

function differentialEntropy(belief::Belief{N}) where { N <: NormalMeanPrecision }
    return 0.5 * log(var(belief)) + 0.5 * log(2 * π) + 0.5
end

function labsgamma(x::Number)
    return SpecialFunctions.logabsgamma(x)[1]
end

function differentialEntropy(belief::Belief{G}) where { G <: GammaAB }
    distribution = getdata(belief)
    return labsgamma(distribution.a) -
    (distribution.a - 1.0) * digamma(distribution.a) -
    log(distribution.b) +
    distribution.a
end

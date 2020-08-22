

function score(::AverageEnergy, ::Type{ <: NormalMeanPrecision }, marginals::Tuple{Marginal, Marginal, Marginal})
    m_mean, v_mean = mean(marginals[1]), var(marginals[1])
    m_out, v_out = mean(marginals[3]), var(marginals[3])

    result = 0.5 * log(2π) -
        0.5 * log(mean(marginals[2])) +
        0.5 * mean(marginals[2]) * (v_out + v_mean + (m_out - m_mean)^2)
    return result
end
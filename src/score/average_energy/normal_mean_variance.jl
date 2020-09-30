
function score(::AverageEnergy, ::Type{ <: NormalMeanVariance }, marginals::Tuple{ Marginal, Marginal, Marginal }, ::Nothing)
    
    m_mean, v_mean = mean(marginals[2]), var(marginals[2])
    m_out, v_out = mean(marginals[1]), var(marginals[1])

    return 0.5 * log(2*pi) + 0.5 * logmean(marginals[3]) + 0.5 * inversemean(marginals[3])*(v_out + v_mean + (m_out - m_mean)^2)
end

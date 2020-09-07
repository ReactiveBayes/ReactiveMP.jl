function score(
    ::AverageEnergy, 
    ::Type{ <: NormalMeanVariance{T} }, 
    marginals::Tuple{ Marginal{ Tuple{T,T,NormalMeanVariance{T}} } },
    ::Nothing) where { T <: Real }
    ##
    return score(AverageEnergy(), NormalMeanVariance{T}, map(as_marginal, getdata(marginals[1])), nothing)
end

function score(
    ::AverageEnergy, 
    ::Type{ <: NormalMeanVariance }, 
    marginals::Tuple{ Marginal{ <: Tuple{NormalMeanVariance{T},T,T} } },
    ::Nothing) where { T <: Real }
    ##
    return score(AverageEnergy(), NormalMeanVariance{T}, map(as_marginal, getdata(marginals[1])), nothing)
end

function score(
    ::AverageEnergy, 
    ::Type{ <: NormalMeanVariance }, 
    marginals::Tuple{Marginal,Marginal,Marginal},
    ::Nothing)
    ##
    m_mean, v_mean = mean(marginals[1]), var(marginals[1])
    m_out, v_out = mean(marginals[3]), var(marginals[3])

    return 0.5*log(2*pi) + 0.5*logmean(marginals[2]) + 0.5*inversemean(marginals[2])*(v_out + v_mean + (m_out - m_mean)^2)
end
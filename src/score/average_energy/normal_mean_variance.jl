function score(
    ::AverageEnergy, 
    ::Type{NormalMeanVariance{Float64}}, 
    marginals::Tuple{Marginal{Tuple{Float64,Float64,NormalMeanVariance{Float64}}}})
    ##
    factorised = map(as_marginal, getdata(marginals[1]))
    return score(AverageEnergy(), NormalMeanVariance{Float64}, factorised)
end

function score(
    ::AverageEnergy, 
    ::Type{ <: NormalMeanVariance }, 
    marginals::Tuple{ Marginal{ <: Tuple{NormalMeanVariance{T},T,T}} }) where { T <: Real }
    ##
    return score(AverageEnergy(), NormalMeanVariance{Float64}, map(as_marginal, getdata(marginals[1])))
end

function score(
    ::AverageEnergy, 
    ::Type{ <: NormalMeanVariance }, 
    marginals::Tuple{Marginal,Marginal,Marginal})
    ##
    m_mean, v_mean = mean(marginals[1]), var(marginals[1])
    m_out, v_out = mean(marginals[3]), var(marginals[3])

    0.5*log(2*pi) +
    0.5*logmean(marginals[2]) +
    0.5*inversemean(marginals[2])*(v_out + v_mean + (m_out - m_mean)^2)
end
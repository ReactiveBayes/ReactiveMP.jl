
function score(
    ::AverageEnergy, 
    ::Type{ <: GammaAB }, 
    marginals::Tuple{Marginal, Marginal, Marginal}, 
    ::Nothing)
    ##
    return labsgamma(mean(marginals[2])) - mean(marginals[2]) * log(mean(marginals[3])) -
        (mean(marginals[2]) - 1.0) * log(mean(marginals[1])) +
        mean(marginals[3]) * mean(marginals[1])
end


function score(
    ::AverageEnergy, 
    ::Type{ <: GammaAB }, 
    marginals::Tuple{ Marginal{ Tuple{GammaAB{T}, T, T} } }, 
    ::Nothing) where { T <: Real }
    ##    
    return score(AverageEnergy(), GammaAB, map(as_marginal, getdata(marginals[1])), nothing)
end

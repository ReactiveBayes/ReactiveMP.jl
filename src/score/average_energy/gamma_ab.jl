
function score(
    ::AverageEnergy, 
    ::Type{ <: GammaAB }, 
    marginals::Tuple{Marginal, Marginal, Marginal}, 
    ::Nothing)
    ##
    return labsgamma(mean(marginals[1])) - mean(marginals[1]) * log(mean(marginals[2])) -
        (mean(marginals[1]) - 1.0) * log(mean(marginals[3])) +
        mean(marginals[2]) * mean(marginals[3])
end


function score(
    ::AverageEnergy, 
    ::Type{ <: GammaAB }, 
    marginals::Tuple{ Marginal{ Tuple{T, T, GammaAB{T}} } }, 
    ::Nothing) where { T <: Real }
    ##    
    return score(AverageEnergy(), GammaAB, map(as_marginal, getdata(marginals[1])), nothing)
end

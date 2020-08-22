function score(::AverageEnergy, ::Type{ <: GammaAB }, marginals::Tuple{Marginal, Marginal, Marginal})
    return labsgamma(mean(marginals[1])) - mean(marginals[1]) * log(mean(marginals[2])) -
    (mean(marginals[1]) - 1.0) * log(mean(marginals[3])) +
    mean(marginals[2]) * mean(marginals[3])
end


function score(::AverageEnergy, ::Type{ <: Gamma }, marginals::Tuple{Marginal, Marginal, Marginal}, ::Nothing)
    return labsgamma(mean(marginals[2])) - mean(marginals[2]) * log(inv(mean(marginals[3]))) -
        (mean(marginals[2]) - 1.0) * log(mean(marginals[1])) + inv(mean(marginals[3])) * mean(marginals[1])
end

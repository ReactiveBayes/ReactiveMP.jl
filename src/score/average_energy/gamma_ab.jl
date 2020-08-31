
function score(::AverageEnergy, ::Type{ <: GammaAB }, marginals::Tuple{Marginal, Marginal, Marginal})
    return labsgamma(mean(marginals[1])) - mean(marginals[1]) * log(mean(marginals[2])) -
    (mean(marginals[1]) - 1.0) * log(mean(marginals[3])) +
    mean(marginals[2]) * mean(marginals[3])
end


function score(::AverageEnergy, ::Type{ <: GammaAB }, marginals::Tuple{ Marginal{ Tuple{T, T, GammaAB{T}} } }) where { T <: Real }
    factorised = getdata(marginals[1])
    return score(AverageEnergy(), GammaAB, (factorised[1] |> as_marginal, factorised[2] |> as_marginal, factorised[3] |> as_marginal))
end

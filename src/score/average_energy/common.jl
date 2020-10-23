
function score(::AverageEnergy, fform, marginals::Tuple{ <: Marginal{ <: Tuple } }, meta)
    return score(AverageEnergy(), fform, map(as_marginal, getdata(marginals[1])), meta)
end

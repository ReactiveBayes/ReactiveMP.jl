
function score(::AverageEnergy, functional_form, marginals::Tuple{ Marginal{ <: Tuple } }, meta)
    return score(AverageEnergy(), functional_form, map(as_marginal, getdata(marginals[1])), meta)
end
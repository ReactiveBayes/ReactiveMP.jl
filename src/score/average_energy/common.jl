
function score(::AverageEnergy, functional_form, marginals::Tuple{ Marginal{ <: FactorizedMarginal } }, meta)
    return score(AverageEnergy(), functional_form, map(as_marginal, marginals[1] |> getdata |> getfactors), meta)
end
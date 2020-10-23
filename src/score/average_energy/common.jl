function score(::AverageEnergy, fform, ::Type{ Val{ N } }, marginals::Tuple{ <: Marginal{ <: Tuple } }, meta) where N
    return score(AverageEnergy(), fform, split_underscored_symbol(Val{ N[1] }), map(as_marginal, getdata(marginals[1])), meta)
end

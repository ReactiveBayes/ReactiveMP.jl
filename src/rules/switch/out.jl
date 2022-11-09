@rule Switch(:out, Marginalisation) (m_switch::Any, m_inputs::ManyOf{N, Any}) where {N} = begin

    # get logscales of different inputs
    logscales_inputs = map(getlogscale, messages[2])

    # get logscales of Categorical/Bernoulli
    logscales_switch = getlogscale(messages[1]) .+ log.(probvec(m_switch))

    # compute logscales of individual components
    logscales = logscales_inputs .+ logscales_switch

    @logscale logsumexp(logscales)

    # compute weights
    w = Categorical(softmax(collect(logscales)))

    # return mixture 
    return MixtureModel(collect(m_inputs), w)
end

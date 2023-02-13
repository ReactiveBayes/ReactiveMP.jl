@rule Switch(:out, Marginalisation) (m_switch::Any, m_inputs::ManyOf{N, Any}) where {N} = begin

    # get logscales of different inputs
    logscales_inputs = map(getlogscale, messages[2])

    # get logscales of Categorical/Bernoulli
    logscales_switch = getlogscale(messages[1]) .+ log.(probvec(m_switch))

    # compute logscales of individual components
    logscales = logscales_inputs .+ logscales_switch

    @logscale logsumexp(logscales)

    # compute weights
    w = softmax(collect(logscales))

    # return mixture 
    return MixtureDistribution(collect(m_inputs), w)
end

@rule Switch(:out, Marginalisation) (m_inputs::ManyOf{N, Any}, q_switch::PointMass) where {N} = begin

    # check whether mean is one-hot
    p = mean(q_switch)
    @assert sum(p) ≈ 1 "The selector variable connected to the switch node is not normalized."
    @assert all(x -> x == 1 || x == 0, p) "The selector variable connected to the switch node is not one-hot encoded."

    # get selected cluster
    kmax = argmax(p)

    @logscale 0
    return m_inputs[kmax]
end

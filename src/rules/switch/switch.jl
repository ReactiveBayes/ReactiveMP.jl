@rule Switch(:switch, Marginalisation) (
    m_out::Any,
    m_inputs::NTuple{N, Any}
) where {N} = begin

    #  compute logscales of different products
    logscales = map(input -> getlogscale(messages[1] * input), messages[2])
    
    @logscale logsumexp(logscales)

    return Categorical(softmax([logscales...]))
end
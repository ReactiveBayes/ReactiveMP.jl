import StaticArrays: SVector

@rule Mixture(:switch, Marginalisation) (m_out::Any, m_inputs::ManyOf{N, Any}) where {N} = begin

    #  compute logscales of different products
    # `messages` are available from the `@rule` macro itself
    @show messages[1]
    @show getlogscale(messages[1])
    logscales = map(input -> getlogscale(messages[1] * input), messages[2])

    @logscale logsumexp(logscales)

    return Categorical(softmax(collect(logscales)))
end

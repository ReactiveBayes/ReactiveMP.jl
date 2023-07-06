import StaticArrays: SVector

@rule Mixture(:switch, Marginalisation) (m_out::Any, m_inputs::ManyOf{N, Any}) where {N} = begin
    #  compute logscales of different products
    # `messages` are available from the `@rule` macro itself
    logscales = map(input -> getlogscale(messages[1] * input), messages[2])

    @logscale logsumexp(logscales)

    return Categorical(softmax(collect(logscales)))
end

@rule Mixture(:switch, Marginalisation) (m_out::PointMass, m_inputs::ManyOf{N, Any}) where {N} = begin
    out = mean(m_out)
    eval_logpdf = map(input -> logpdf(input,out), m_inputs)
    logscales = map(input -> getlogscale(input), messages[2])
    newlogscale = logscales .+ eval_logpdf

    @logscale logsumexp(newlogscale)
    
    return Categorical(softmax(collect(newlogscale)))
end
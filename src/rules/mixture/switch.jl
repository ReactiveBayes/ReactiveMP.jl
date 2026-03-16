import StaticArrays: SVector

@rule Mixture(:switch, Marginalisation) (
    m_out::Any, m_inputs::ManyOf{N, Any}
) where {N} = begin

    #  compute logscales of different products
    # `messages` are available from the `@rule` macro itself
    logscales = map(
        input -> getlogscale(
            compute_product_of_messages(
                MessageProductContext(), messages[1], input
            )
        ),
        messages[2]
    )

    @logscale logsumexp(logscales)

    return Categorical(softmax(collect(logscales)))
end

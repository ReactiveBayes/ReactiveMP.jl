@rule Mixture(:switch, Marginalisation) (
    m_out::Any, m_inputs::ManyOf{N, Any}
) where {N} = begin

    #  compute logscales of different products
    # `messages` are available from the `@rule` macro itself
    logscales = map(
        input -> getlogscale(
            ReactiveMP.getannotations(
                compute_product_of_two_messages(
                    ReactiveMP.randomvar(; label = :mixture_switch_rule),
                    ReactiveMP.MessageProductContext(;
                        annotations = (LogScaleAnnotations(),)
                    ),
                    messages[1],
                    input,
                ),
            ),
        ),
        messages[2],
    )

    @logscale logsumexp(logscales)

    return Categorical(softmax(collect(logscales)))
end

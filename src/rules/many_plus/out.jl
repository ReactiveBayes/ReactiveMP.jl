@rule ManyPlus(:out, Marginalisation) (
    m_inputs::ManyOf{N, UnivariateNormalDistributionsFamily},
) where {N} = begin
    input_mean, input_variance = mean_var(first(m_inputs))
    output_mean, output_variance = foldl(
        (acc, message) -> begin
            message_mean, message_variance = mean_var(message)
            return acc[1] + message_mean, acc[2] + message_variance
        end,
        m_inputs,
        init = (zero(input_mean), zero(input_variance)),
    )
    return NormalMeanVariance(output_mean, output_variance)
end

# `m_inputs` contains only the other inputs; the target is excluded by the node.
@rule ManyPlus((:inputs, k), Marginalisation) (
    m_out::UnivariateNormalDistributionsFamily,
    m_inputs::ManyOf{
        N, Union{UnivariateNormalDistributionsFamily, PointMass{<:Real}}
    },
) where {N} = begin
    output_mean, output_variance = mean_var(m_out)
    other_mean, other_variance = foldl(
        (acc, message) -> begin
            message_mean, message_variance = mean_var(message)
            return acc[1] + message_mean, acc[2] + message_variance
        end,
        m_inputs,
        init = (zero(output_mean), zero(output_variance)),
    )
    return NormalMeanVariance(
        output_mean - other_mean, output_variance + other_variance
    )
end

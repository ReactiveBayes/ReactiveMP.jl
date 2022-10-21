@rule Switch(:out, Marginalisation) (
    m_switch::Any,
    m_inputs::Tuple{Any}
) where {N} = begin
    @logscale 1
    return Beta(1, 1)
end
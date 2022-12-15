export rule

@rule Dirichlet(:out, Marginalisation) (m_a::PointMass{<:AbstractVector},) = begin
    @logscale 0
    return Dirichlet(mean(m_a))
end

@rule Dirichlet(:out, Marginalisation) (q_a::PointMass{<:AbstractVector},) = begin
    @logscale 0
    return Dirichlet(mean(q_a))
end

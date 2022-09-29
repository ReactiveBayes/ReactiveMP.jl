export rule

@rule Categorical(:out, Marginalisation) (q_p::Dirichlet,) = begin
    rho = clamp.(exp.(mean(log, q_p)), tiny, Inf) # Softens the parameter
    return Categorical(rho ./ sum(rho))
end

@rule Categorical(:out, Marginalisation) (m_p::PointMass,) = begin
    @logscale 0
    return Categorical(mean(m_p))
end

@rule Categorical(:out, Marginalisation) (q_p::PointMass,) = begin
    @logscale 0
    return Categorical(mean(q_p))
end

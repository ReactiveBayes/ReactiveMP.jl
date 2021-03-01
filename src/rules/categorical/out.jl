export rule

@rule Categorical(:out, Marginalisation) (q_p::Dirichlet, ) = begin
    rho = clamp.(exp.(logmean(q_p)), tiny, Inf) # Softens the parameter
    return Categorical(rho ./ sum(rho))
end

@rule Categorical(:out, Marginalisation) (m_p::PointMass, ) = begin
    return Categorical(mean(m_p))
end

@rule Categorical(:out, Marginalisation) (q_p::PointMass, ) = begin
    return Categorical(mean(q_p))
end

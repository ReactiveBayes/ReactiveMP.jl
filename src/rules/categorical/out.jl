export rule

@rule Categorical(:out, Marginalisation) (m_p::Dirichlet,) = begin
    if isnothing(messages[1].addons)
        @logscale 0
    else 
        @logscale getlogscale(messages[1])
    end
    return Categorical(mean(m_p))
end

@rule Categorical(:out, Marginalisation) (q_p::Dirichlet,) = begin
    rho = clamp.(exp.(mean(log, q_p)), tiny, Inf) # Softens the parameter
    return Categorical(rho ./ sum(rho))
end

@rule Categorical(:out, Marginalisation) (m_p::PointMass,) = Categorical(mean(m_p))

@rule Categorical(:out, Marginalisation) (q_p::PointMass,) = Categorical(mean(q_p))

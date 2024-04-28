import Base.Broadcast: BroadcastFunction

@rule Categorical(:out, Marginalisation) (m_p::Dirichlet,) = begin
    @logscale 0
    return Categorical(mean(m_p))
end

@rule Categorical(:out, Marginalisation) (q_p::Dirichlet,) = begin
    rho = clamp.(exp.(mean(BroadcastFunction(log), q_p)), tiny, Inf) # Softens the parameter
    return Categorical(rho ./ sum(rho))
end

@rule Categorical(:out, Marginalisation) (m_p::PointMass,) = Categorical(mean(m_p))

@rule Categorical(:out, Marginalisation) (q_p::PointMass,) = Categorical(mean(q_p))

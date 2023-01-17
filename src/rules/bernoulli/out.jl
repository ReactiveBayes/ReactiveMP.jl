export rule

@rule Bernoulli(:out, Marginalisation) (m_p::Beta,) = begin
    @logscale 0
    return Bernoulli(mean(m_p))
end

@rule Bernoulli(:out, Marginalisation) (m_p::PointMass,) = Bernoulli(mean(m_p))

@rule Bernoulli(:out, Marginalisation) (q_p::PointMass,) = Bernoulli(mean(q_p))

@rule Bernoulli(:out, Marginalisation) (q_p::Beta,) = begin
    rho_1 = clamp(exp(mean(log, q_p)), tiny, huge)
    rho_2 = clamp(exp(mean(mirrorlog, q_p)), tiny, huge)
    return Bernoulli(rho_1 / (rho_1 + rho_2))
end

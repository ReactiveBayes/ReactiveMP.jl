export rule

@rule Bernoulli(:out, Marginalisation) (m_p::Beta,) = begin
    @logscale 0
    return Bernoulli(mean(m_p))
end

@rule Bernoulli(:out, Marginalisation) (m_p::PointMass,) = Bernoulli(mean(m_p))

@rule Bernoulli(:out, Marginalisation) (q_p::PointMass,) = Bernoulli(mean(q_p))

# q_p::Any s.t. domain = [0, 1]
@rule Bernoulli(:out, Marginalisation) (q_p::Any,) = begin
    rho_1 = mean(log, q_p)          # E[ln(x)]
    rho_2 = mean(mirrorlog, q_p)    # E[log(1-x)]
    m = max(rho_1, rho_2)
    tmp = exp(rho_1 - m)
    p = clamp(tmp / (tmp + exp(rho_2 - m)), tiny, one(m))
    return Bernoulli(p)
end

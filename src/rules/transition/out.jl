import Base.Broadcast: BroadcastFunction

# Belief Propagation                #
# --------------------------------- #

@rule Transition(:out, Marginalisation) (m_in::Union{PointMass, DiscreteNonParametric}, m_a::PointMass) = begin
    @logscale 0
    p = mean(m_a) * probvec(m_in)
    normalize!(p, 1)
    return Categorical(p)
end

@rule Transition(:out, Marginalisation) (q_in::PointMass, q_a::PointMass) = begin
    @logscale 0
    p = mean(q_a) * mean(q_in)
    normalize!(p, 1)
    return Categorical(p)
end

# Variational                       # 
# --------------------------------- #

@rule Transition(:out, Marginalisation) (q_in::DiscreteNonParametric, q_a::Any) = begin
    a = clamp.(exp.(mean(BroadcastFunction(log), q_a) * probvec(q_in)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:out, Marginalisation) (m_in::DiscreteNonParametric, q_a::ContinuousMatrixDistribution) = begin
    a = clamp.(exp.(mean(BroadcastFunction(log), q_a)) * probvec(m_in), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:out, Marginalisation) (m_in::Union{PointMass, DiscreteNonParametric}, q_a::PointMass, meta::Any) = begin
    @logscale 0
    return @call_rule Transition(:out, Marginalisation) (m_in = m_in, m_a = q_a, meta = meta, addons = getaddons())
end


@rule Transition(:in, Marginalisation) (m_out::Categorical, m_a::PointMass) = begin
    @logscale log(sum(mean(A)' * probvec(m_out)))
    p = mean(m_a)' * probvec(m_out)
    normalize!(p, 1)
    return Categorical(p)
end

@rule Transition(:in, Marginalisation) (q_out::Any, q_a::MatrixDirichlet) = begin
    a = clamp.(exp.(mean(log, q_a)' * probvec(q_out)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:in, Marginalisation) (m_out::Categorical, q_a::MatrixDirichlet) = begin
    a = clamp.(exp.(mean(log, q_a))' * probvec(m_out), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:in, Marginalisation) (m_out::Categorical, q_a::PointMass, meta::Any) = begin
    return @call_rule Transition(:in, Marginalisation) (m_out = m_out, m_a = q_a, meta = meta)
end

@rule Transition(:in, Marginalisation) (q_out::PointMass, q_a::PointMass) = begin
    p = mean(q_a)' * mean(q_out)
    normalize!(p, 1)
    return Categorical(p)
end

### PAD experiment
@rule Transition(:in, Marginalisation) (m_out::DiscreteNonParametric, q_out::PointMass, q_a::MatrixDirichlet, ) = begin 
    #### scalefactor
    # if isnothing(messages[1].addons)
    #     @logscale 0
    # else
    #     A = mean(q_a)
    #     y = mean(q_out)
    #     @logscale log(sum(A'*y)) + getlogscale(messages[1]) #might be incorrect 
    # end
    ####
    @logscale 0 #might be incorrect 
    a = clamp.(exp.(mean(log, q_a)' * probvec(q_out)), tiny, Inf)
    return Categorical(a ./ sum(a))
end
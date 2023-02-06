export rule

@rule Bernoulli(:p, Marginalisation) (m_out::PointMass,) = begin
    @logscale -log(2)
    r = mean(m_out)
    Beta(one(r) + r, 2one(r) - r)
end

@rule Bernoulli(:p, Marginalisation) (q_out::PointMass,) = begin
    @logscale -log(2)
    r = mean(q_out)
    Beta(one(r) + r, 2one(r) - r)
end

@rule Bernoulli(:p, Marginalisation) (q_out::Bernoulli,) = begin
    r = succprob(q_out)
    return Beta(one(r) + r, 2one(r) - r)
end

@rule Bernoulli(:p, Marginalisation) (q_out::Categorical,) = begin
    p = probvec(q_out)
    @assert length(p) == 2 "Bernoulli is only defined over its support {0,1}. It has received a Categorical message containing a probability vector unequal to 2."
    r = p[2]
    return Beta(one(r) + r, 2one(r) - r)
end

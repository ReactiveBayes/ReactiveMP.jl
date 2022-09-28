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
    r = first(probvec(q_out))
    return Beta(one(r) + r, 2one(r) - r)
end

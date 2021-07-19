import StatsFuns: normpdf, normcdf

@rule Probit(:out, Marginalisation) (m_in::PointMass, ) = begin
    p = normcdf(mean(m_in))
    return Bernoulli(p)
end

@rule Probit(:out, Marginalisation) (m_in::UnivariateNormalDistributionsFamily, ) = begin
    p = normcdf(mean(m_in)/sqrt(1+var(m_in)))
    return Bernoulli(p)
end

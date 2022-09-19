import StatsFuns: normpdf, normcdf

@rule Probit(:out, Marginalisation) (m_in::PointMass, meta::Union{ProbitMeta, Nothing}) = begin
    p = normcdf(mean(m_in))
    return Bernoulli(p)
end

@rule Probit(:out, Marginalisation) (q_in::PointMass, meta::Union{ProbitMeta, Nothing}) = begin
    p = normcdf(mean(q_in))
    return Bernoulli(p)
end

@rule Probit(:out, Marginalisation) (m_in::UnivariateNormalDistributionsFamily, meta::Union{ProbitMeta, Nothing}) =
    begin
        p = normcdf(mean(m_in) / sqrt(1 + var(m_in)))
        return Bernoulli(p)
    end

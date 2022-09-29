
# Belief Propagation                #
# --------------------------------- #
@rule MvNormalMeanPrecision(:μ, Marginalisation) (m_out::PointMass, m_Λ::PointMass) = begin
    @logscale 0
    return MvNormalMeanPrecision(mean(m_out), mean(m_Λ))
end

@rule MvNormalMeanPrecision(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_Λ::PointMass) = begin
    @logscale 0
    m_out_mean, m_out_cov = mean_cov(m_out)
    return MvNormalMeanCovariance(m_out_mean, m_out_cov + cholinv(mean(m_Λ)))
end

# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanPrecision(:μ, Marginalisation) (q_out::Any, q_Λ::Any) = MvNormalMeanPrecision(mean(q_out), mean(q_Λ))

@rule MvNormalMeanPrecision(:μ, Marginalisation) (m_out::PointMass, q_Λ::Any) =
    MvNormalMeanPrecision(mean(m_out), mean(q_Λ))

@rule MvNormalMeanPrecision(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, q_Λ::Any) = begin
    m_out_mean, m_out_cov = mean_cov(m_out)
    return MvNormalMeanCovariance(m_out_mean, m_out_cov + cholinv(mean(q_Λ)))
end

@rule MvNormalMeanPrecision(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, q_Λ::Wishart) = begin
    m_out_mean, m_out_cov = mean_cov(m_out)
    inv_mean_q_Λ = inv(q_Λ.S.chol) ./ q_Λ.df
    return MvNormalMeanCovariance(m_out_mean, m_out_cov + inv_mean_q_Λ)
end

import ForwardDiff: hessian
# Belief Propagation                #
# --------------------------------- #
@rule NormalMeanVariance(:μ, Marginalisation) (m_out::PointMass, m_v::PointMass) = NormalMeanVariance(mean(m_out), mean(m_v))

@rule NormalMeanVariance(:μ, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_v::PointMass) = begin
    if isnothing(messages[1].addons)
        @logscale 0
    else 
        @logscale getlogscale(messages[1])
    end
    m_out_mean, m_out_cov = mean_cov(m_out)
    return NormalMeanVariance(m_out_mean, m_out_cov + mean(m_v))
end

# Variational                       # 
# --------------------------------- #
@rule NormalMeanVariance(:μ, Marginalisation) (q_out::PointMass, q_v::PointMass) = NormalMeanVariance(mean(q_out), mean(q_v))

@rule NormalMeanVariance(:μ, Marginalisation) (q_out::Any, q_v::Any) = NormalMeanVariance(mean(q_out), mean(q_v))

@rule NormalMeanVariance(:μ, Marginalisation) (m_out::PointMass, q_v::Any) = NormalMeanVariance(mean(m_out), mean(q_v))

@rule NormalMeanVariance(:μ, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, q_v::Any) = begin
    if isnothing(messages[1].addons)
        @logscale 0
    else 
        @logscale getlogscale(messages[1])
    end
    m_out_mean, m_out_cov = mean_cov(m_out)
    return NormalMeanVariance(m_out_mean, m_out_cov + mean(q_v))
end

## gp test 
# @rule NormalMeanVariance(:μ, Marginalisation) (m_out::ContinuousUnivariateLogPdf, q_v::PointMass, ) = begin 
#     res = optimize(x -> -logpdf(m_out,x), -10,10)
#     m0 = res.minimizer
#     # v0 = - cholinv(hessian(x -> logpdf(m_out,x), [m0[1]]))
#     # @show v0
#     v0 = 0.1
#     l_pdf(x) = logpdf(m_out,x) - logpdf(d0,x)
#     meta = GaussHermiteCubature(71)
#     m,v = ReactiveMP.approximate_meancov(meta, z -> exp(l_pdf(z)), m0[1], v0)

#     d = NormalMeanVariance(m0[1],v0[1])

#     return @call_rule NormalMeanVariance(:μ, Marginalisation) (m_out=d, q_v=q_v)
# end

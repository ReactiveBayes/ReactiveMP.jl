export rule

@rule Poisson(:out, Marginalisation) (m_l::PointMass,) = Poisson(mean(m_l))

@rule Poisson(:out, Marginalisation) (q_l::GammaDistributionsFamily,) = Poisson(exp(digamma(shape(q_l))) / rate(q_l))

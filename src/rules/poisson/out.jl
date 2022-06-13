export rule

@rule Poisson(:out, Marginalisation) (m_l::PointMass,) = Poisson(mean(m_l))

@rule Poisson(:out, Marginalisation) (q_l::GammaDistributionsFamily,) = Poisson((shape(q_l) - 1//2)/rate(q_l))
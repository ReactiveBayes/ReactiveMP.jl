@rule(
    form        => Type{ <: MvNormalMeanCovariance },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_mean::Dirac, m_covariance::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(m_mean), mean(m_covariance))
    end
)

@rule(
    form        => Type{ <: MvNormalMeanCovariance },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_mean::Any, q_covariance::Any),
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(q_mean), mean(q_covariance))
    end
)

@rule(
    form        => Type{ <: MvNormalMeanCovariance },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_mean::MvNormalMeanCovariance, ),
    marginals   => (q_covariance::Any, ),
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(m_mean), cov(m_mean) + mean(q_covariance))
    end
)
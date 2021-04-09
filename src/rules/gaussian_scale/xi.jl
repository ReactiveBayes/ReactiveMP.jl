export rule

@rule GS(:xi, Marginalisation) (q_X::Any, ) = begin

    mean_X, var_X = mean(q_X), real(var(q_X))

    m_xi = log(var_X + abs2(mean_X))

    return GaussianMeanPrecision(m_xi, 1)
end
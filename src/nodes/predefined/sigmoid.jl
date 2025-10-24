using StatsFuns: logistic, softplus
using Distributions: pdf

export Sigmoid

struct Sigmoid end

@node Sigmoid Stochastic [out, in, ζ]

@average_energy Sigmoid (q_out::Categorical, q_in::UnivariateNormalDistributionsFamily, q_ζ::PointMass) = begin
    m_out = pdf(q_out, 1)
    m_in, v_in = mean_var(q_in)

    ζ_hat = mean(q_ζ)

    U = -(m_in * m_out - softplus(-ζ_hat) - (0.5 * (m_in + ζ_hat)) - 0.5 * ((logistic(ζ_hat) - 0.5)/ζ_hat) * (m_in^2 + v_in - ζ_hat^2))
    return U
end

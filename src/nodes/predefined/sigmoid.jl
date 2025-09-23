struct Sigmoid end
using StatsFuns: logistic

@node Sigmoid Stochastic [out, in, ξ]

@average_energy Sigmoid (q_out::Categorical, q_in::UnivariateNormalDistributionsFamily, q_ξ::PointMass) = begin
    
    mout = mean(q_out)
    m_in, v_in = mean_var(q_in)
    
    ξ_hat = mean(q_ξ)

    U = m_in * mout + log(logistic(ξ_hat)) - 0.5 * (m_in + ξ_hat) - ((logistic(ξ_hat) - 0.5)/(2*ξ_hat)) * (m_in^2 + v_in - ξ_hat^2)
    return U
end

export Sigmoid
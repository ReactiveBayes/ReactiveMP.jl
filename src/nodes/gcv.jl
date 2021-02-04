export GCV

struct GCV end

@node GCV Stochastic [ y, x, z, κ, ω ]

@average_energy GCV (q_y_x::MvNormalMeanCovariance, q_z::NormalMeanVariance, q_κ::PointMass, q_ω::PointMass) = begin
    m_yx, cov_yx = mean(q_y_x), cov(q_y_x)
    m_z, var_z = mean(q_z), var(q_z)
    m_κ, var_κ = mean(q_κ), var(q_κ)
    m_ω, var_ω = mean(q_κ), var(q_κ)

    ksi = (m_κ^2)*var_z + (m_z^2)*var_κ + var_κ*var_z
    psi = (m_yx[1]-m_yx[2])^2 + cov_yx[2,2]+cov_yx[1,1]-cov_yx[2,1]-cov_yx[1,2]
    A = exp(-m_ω + var_ω/2)
    B = exp(-m_κ*m_z + ksi/2)

    return 0.5 * log(2pi) + 0.5 * (m_z*m_κ+m_ω) + 0.5 * (psi*A*B)
end
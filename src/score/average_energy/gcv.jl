function score(
    ::AverageEnergy,
    ::Type{GCV}, 
    marginals::Tuple{
        Marginal{ <: MvNormalMeanCovariance{T} },
        Marginal{ <: NormalMeanVariance{T} },
        Marginal{T},
        Marginal{T}
    },
    ::Nothing) where { T <: Real }
    ##
    m_yx, cov_yx = mean(marginals[1]), cov(marginals[1])
    m_z, var_z = mean(marginals[2]), var(marginals[2])
    m_κ, var_κ = mean(marginals[3]), var(marginals[3])
    m_ω, var_ω = mean(marginals[4]), var(marginals[4])

    ksi = (m_κ^2)*var_z + (m_z^2)*var_κ + var_κ*var_z
    psi = (m_yx[1]-m_yx[2])^2 + cov_yx[2,2]+cov_yx[1,1]-cov_yx[2,1]-cov_yx[1,2]
    A = exp(-m_ω + var_ω/2)
    B = exp(-m_κ*m_z + ksi/2)

    return 0.5 * log(2pi) + 0.5 * (m_z*m_κ+m_ω) + 0.5 * (psi*A*B)
end
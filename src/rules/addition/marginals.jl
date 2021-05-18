export marginalrule

@marginalrule typeof(+)(:in1_in2) (m_out::NormalMeanVariance, m_in1::NormalMeanVariance, m_in2::PointMass) = begin
    (in1 = prod(ProdPreserveParametrisation(), NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out)), m_in1), in2 = m_in2)
end

@marginalrule typeof(+)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = begin
    W_out  = real(invcov(m_out))
    xi_out = W_out * mean(m_out)

    W_in1  = real(invcov(m_in1))
    xi_in1 = W_in1 * mean(m_in1)

    W_in2  = real(invcov(m_in2))
    xi_in2 = W_in2 * mean(m_in2)
    
    xi = [ xi_in1 + xi_out; xi_in2 + xi_out ]
    W  = [ W_in1+W_out W_out; W_out W_in2+W_out ]
    
    Σ = cholinv(W)
    μ = Σ * xi


    return MvNormalMeanPrecision(μ, W)
end

@marginalrule typeof(+)(:in1_in2) (m_out::ComplexNormal, m_in1::ComplexNormal, m_in2::PointMass) = begin
    (in1 = prod(ProdPreserveParametrisation(), ComplexNormal(mean(m_out) - mean(m_in2), var(m_out)), m_in1), in2 = m_in2)
end

@marginalrule typeof(+)(:in1_in2) (m_out::ComplexNormal, m_in1::ComplexNormal, m_in2::ComplexNormal) = begin
    W_out  = real(invcov(m_out))
    xi_out = W_out * mean(m_out)

    W_in1  = real(invcov(m_in1))
    xi_in1 = W_in1 * mean(m_in1)

    W_in2  = real(invcov(m_in2))
    xi_in2 = W_in2 * mean(m_in2)
    
    xi = [ xi_in1 + xi_out; xi_in2 + xi_out ]
    W  = [ W_in1+W_out W_out; W_out W_in2+W_out ]
    
    Σ = cholinv(W)
    μ = Σ * xi


    return MvComplexNormal(μ, Σ, 0)
end
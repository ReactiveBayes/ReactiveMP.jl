
@marginalrule typeof(+)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    mout, vout = mean_var(m_out)
    return (in1 = prod(ProdAnalytical(), NormalMeanVariance(mout - mean(m_in2), vout), m_in1), in2 = m_in2)
end

@marginalrule typeof(+)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass, m_in2::UnivariateNormalDistributionsFamily) = begin
    mout, vout = mean_var(m_out)
    return (in1 = m_in1, in2 = prod(ProdAnalytical(), NormalMeanVariance(mout - mean(m_in1), vout), m_in2))
end

@marginalrule typeof(+)(:in1_in2) (m_out::MvNormalMeanCovariance, m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    mout, Vout = mean_cov(m_out)
    return (in1 = prod(ProdAnalytical(), MvNormalMeanCovariance(mout - mean(m_in2), Vout), m_in1), in2 = m_in2)
end

@marginalrule typeof(+)(:in1_in2) (m_out::MvNormalMeanPrecision, m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    mout, Wout = mean_precision(m_out)
    return (in1 = prod(ProdAnalytical(), MvNormalMeanPrecision(mout - mean(m_in2), Wout), m_in1), in2 = m_in2)
end

@marginalrule typeof(+)(:in1_in2) (m_out::MvNormalWeightedMeanPrecision, m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    xiout, Wout = weightedmean_precision(m_out)
    tmp = -Wout*mean(m_in2)
    tmp .+= xiout
    return (in1 = prod(ProdAnalytical(), MvNormalWeightedMeanPrecision(tmp, Wout), m_in1), in2 = m_in2)
end

@marginalrule typeof(+)(:in1_in2) (m_out::MvNormalMeanCovariance, m_in1::PointMass, m_in2::MultivariateNormalDistributionsFamily) = begin
    mout, Vout = mean_cov(m_out)
    return (in1 = m_in1, in2 = prod(ProdAnalytical(), MvNormalMeanCovariance(mout - mean(m_in1), Vout), m_in2))
end

@marginalrule typeof(+)(:in1_in2) (m_out::MvNormalMeanPrecision, m_in1::PointMass, m_in2::MultivariateNormalDistributionsFamily) = begin
    mout, Wout = mean_precision(m_out)
    return (in1 = m_in1, in2 = prod(ProdAnalytical(), MvNormalMeanPrecision(mout - mean(m_in1), Wout), m_in2))
end

@marginalrule typeof(+)(:in1_in2) (m_out::MvNormalWeightedMeanPrecision, m_in1::PointMass, m_in2::MultivariateNormalDistributionsFamily) = begin
    xiout, Wout = weightedmean_precision(m_out)
    tmp = -Wout * mean(m_in1)
    tmp .+= xiout
    return (in1 = m_in1, in2 = prod(ProdAnalytical(), MvNormalWeightedMeanPrecision(tmp, Wout), m_in2))
end

@marginalrule typeof(+)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = begin
    xi_out, W_out = weightedmean_precision(m_out)
    xi_in1, W_in1 = weightedmean_precision(m_in1)
    xi_in2, W_in2 = weightedmean_precision(m_in2)
    
    return MvNormalWeightedMeanPrecision([ xi_in1 + xi_out; xi_in2 + xi_out ], [ W_in1 + W_out W_out; W_out W_in2 + W_out])
end

@marginalrule typeof(+)(:in1_in2) (m_out::MultivariateNormalDistributionsFamily, m_in1::MultivariateNormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily) = begin
    xi_out, W_out = weightedmean_precision(m_out)
    xi_in1, W_in1 = weightedmean_precision(m_in1)
    xi_in2, W_in2 = weightedmean_precision(m_in2)

    T = promote_type(eltype(W_out), eltype(W_in1), eltype(W_in2))
    d = length(xi_out)
    Λ = Matrix{T}(undef, (2*d, 2*d))
    @inbounds for k2 in 1:d
        @inbounds for k1 in 1:d
            tmp = W_out[k1,k2]
            k1d = k1+d
            k2d = k2+d
            Λ[k1,k2] = tmp + W_in1[k1,k2]
            Λ[k1d,k2] = tmp
            Λ[k1,k2d] = tmp
            Λ[k1d,k2d] = tmp + W_in2[k1,k2]
        end
    end

    xi = Vector{T}(undef, 2*d)
    @inbounds for k = 1:d
        tmp = xi_out[k]
        xi[k]   = tmp + xi_in1[k]
        xi[k+d] = tmp + xi_in2[k]
    end
    
    # naive: return MvNormalWeightedMeanPrecision([ xi_in1 + xi_out; xi_in2 + xi_out ], [ W_in1 + W_out W_out; W_out W_in2 + W_out])
    return MvNormalWeightedMeanPrecision(xi, Λ)
end
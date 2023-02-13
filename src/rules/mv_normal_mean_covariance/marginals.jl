export marginalrule

@marginalrule MvNormalMeanCovariance(:out_μ_Σ) (m_out::MultivariateNormalDistributionsFamily, m_μ::PointMass, m_Σ::PointMass) = begin
    return (out = prod(ProdAnalytical(), MvNormalMeanCovariance(mean(m_μ), mean(m_Σ)), m_out), μ = m_μ, Σ = m_Σ)
end

@marginalrule MvNormalMeanCovariance(:out_μ_Σ) (m_out::PointMass, m_μ::MultivariateNormalDistributionsFamily, m_Σ::PointMass) = begin
    return (out = m_out, μ = prod(ProdAnalytical(), m_μ, MvNormalMeanCovariance(mean(m_out), mean(m_Σ))), Σ = m_Σ)
end

@marginalrule MvNormalMeanCovariance(:out_μ_Σ) (m_out::MultivariateNormalDistributionsFamily, m_μ::MultivariateNormalDistributionsFamily, m_Σ::PointMass) = begin
    xi_out, W_out = weightedmean_precision(m_out)
    xi_m, W_m = weightedmean_precision(m_μ)

    W_bar = cholinv(mean(m_Σ))

    xi = [xi_out; xi_m]

    T = promote_samplefloattype(m_out, m_μ, m_Σ)
    d = length(xi_out)
    W = Matrix{T}(undef, (2 * d, 2 * d))
    @inbounds for k2 in 1:d
        for k1 in 1:d
            tmp1 = W_bar[k1, k2]
            tmp2 = -tmp1
            k1d = k1 + d
            k2d = k2 + d
            W[k1, k2] = tmp1 + W_out[k1, k2]
            W[k1d, k2] = tmp2
            W[k1, k2d] = tmp2
            W[k1d, k2d] = tmp1 + W_m[k1, k2]
        end
    end

    # naive:
    # xi = [ xi_out; xi_m ]
    # W  = [ W_out + W_bar -W_bar; -W_bar W_m + W_bar ]

    return (out_μ = MvNormalWeightedMeanPrecision(xi, W), Σ = m_Σ)
end

@marginalrule MvNormalMeanCovariance(:out_μ) (m_out::MultivariateNormalDistributionsFamily, m_μ::MultivariateNormalDistributionsFamily, q_Σ::Any) = begin
    xi_out, W_out = weightedmean_precision(m_out)
    xi_m, W_m = weightedmean_precision(m_μ)

    W_bar = cholinv(mean(q_Σ))

    xi = [xi_out; xi_m]

    T = promote_samplefloattype(m_out, m_μ, q_Σ)
    d = length(xi_out)
    W = Matrix{T}(undef, (2 * d, 2 * d))
    @inbounds for k2 in 1:d
        for k1 in 1:d
            tmp1 = W_bar[k1, k2]
            tmp2 = -tmp1
            k1d = k1 + d
            k2d = k2 + d
            W[k1, k2] = tmp1 + W_out[k1, k2]
            W[k1d, k2] = tmp2
            W[k1, k2d] = tmp2
            W[k1d, k2d] = tmp1 + W_m[k1, k2]
        end
    end

    # naive:
    # xi = [ xi_out; xi_m ]
    # W  = [ W_out + W_bar -W_bar; -W_bar W_m + W_bar ]

    return MvNormalWeightedMeanPrecision(xi, W)
end

@marginalrule MvNormalMeanCovariance(:out_μ) (m_out::PointMass, m_μ::MultivariateNormalDistributionsFamily, q_Σ::Any) = begin
    return (out = m_out, μ = prod(ProdAnalytical(), MvNormalMeanCovariance(mean(m_out), mean(q_Σ)), m_μ))
end

@marginalrule MvNormalMeanCovariance(:out_μ) (m_out::MultivariateNormalDistributionsFamily, m_μ::PointMass, q_Σ::Any) = begin
    return (out = prod(ProdAnalytical(), MvNormalMeanCovariance(mean(m_μ), mean(q_Σ)), m_out), μ = m_μ)
end

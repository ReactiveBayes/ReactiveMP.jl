export marginalrule

@marginalrule MvNormalMeanScaleMatrixPrecision(:out_μ_γ_G) (m_out::MultivariateNormalDistributionsFamily, m_μ::PointMass, m_γ::PointMass, m_G::PointMass) = begin
    return (out = prod(ClosedProd(), MvNormalMeanPrecision(mean(m_μ), mean(m_γ) * mean(m_G)), m_out), μ = m_μ, γ = m_γ, G = m_G)
end

@marginalrule MvNormalMeanScaleMatrixPrecision(:out_μ_γ_G) (m_out::PointMass, m_μ::MultivariateNormalDistributionsFamily, m_γ::PointMass, m_G::PointMass) = begin
    return (out = m_out, μ = prod(ClosedProd(), m_μ, MvNormalMeanPrecision(mean(m_out), mean(m_γ) * mean(m_G))), γ = m_γ, G = m_G)
end

@marginalrule MvNormalMeanScaleMatrixPrecision(:out_μ_γ_G) (m_out::MultivariateNormalDistributionsFamily, m_μ::MultivariateNormalDistributionsFamily, m_γ::PointMass, m_G::PointMass) = begin
    xi_y, W_y = weightedmean_precision(m_out)
    xi_m, W_m = weightedmean_precision(m_μ)

    W_bar = mean(m_γ) * mean(m_G)

    T = promote_samplefloattype(m_out, m_μ, m_γ, m_G)
    d = length(xi_y)
    Λ = Matrix{T}(undef, (2 * d, 2 * d))
    @inbounds for k2 in 1:d
        @inbounds for k1 in 1:d
            tmp1 = W_bar[k1, k2]
            tmp2 = -tmp1
            k1d = k1 + d
            k2d = k2 + d
            Λ[k1, k2] = tmp1 + W_y[k1, k2]
            Λ[k1d, k2] = tmp2
            Λ[k1, k2d] = tmp2
            Λ[k1d, k2d] = tmp1 + W_m[k1, k2]
        end
    end

    ξ = [xi_y; xi_m]

    return (out_μ = MvNormalWeightedMeanPrecision(ξ, Λ), γ = m_γ, G = m_G)
end

@marginalrule MvNormalMeanScaleMatrixPrecision(:out_μ) (m_out::MultivariateNormalDistributionsFamily, m_μ::MultivariateNormalDistributionsFamily, q_γ::Any, q_G::Any) = begin
    xi_y, W_y = weightedmean_precision(m_out)
    xi_m, W_m = weightedmean_precision(m_μ)

    W_bar = mean(q_γ) * mean(q_G)

    T = promote_samplefloattype(m_out, m_μ, q_γ, q_G)
    d = length(xi_y)
    Λ = Matrix{T}(undef, (2 * d, 2 * d))
    @inbounds for k2 in 1:d
        @inbounds for k1 in 1:d
            tmp1 = W_bar[k1, k2]
            tmp2 = -tmp1
            k1d = k1 + d
            k2d = k2 + d
            Λ[k1, k2] = tmp1 + W_y[k1, k2]
            Λ[k1d, k2] = tmp2
            Λ[k1, k2d] = tmp2
            Λ[k1d, k2d] = tmp1 + W_m[k1, k2]
        end
    end

    ξ = [xi_y; xi_m]

    return MvNormalWeightedMeanPrecision(ξ, Λ)
end

@marginalrule MvNormalMeanScaleMatrixPrecision(:out_μ) (m_out::PointMass, m_μ::MultivariateNormalDistributionsFamily, q_γ::Any, q_G::Any) = begin
    return (out = m_out, μ = prod(ClosedProd(), MvNormalMeanPrecision(mean(m_out), mean(q_γ) * mean(q_G)), m_μ))
end

@marginalrule MvNormalMeanScaleMatrixPrecision(:out_μ) (m_out::MultivariateNormalDistributionsFamily, m_μ::PointMass, q_γ::Any, q_G::Any) = begin
    return (out = prod(ClosedProd(), MvNormalMeanPrecision(mean(m_μ), mean(q_γ) * mean(q_G)), m_out), μ = m_μ)
end

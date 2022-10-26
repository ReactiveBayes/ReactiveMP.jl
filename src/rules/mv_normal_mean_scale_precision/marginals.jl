export marginalrule

@marginalrule MvNormalMeanScalePrecision(:out_μ_γ) (
    m_out::MultivariateNormalDistributionsFamily,
    m_μ::PointMass,
    m_γ::PointMass
) = begin
    return (
        out = prod(
            ProdAnalytical(),
            MvNormalMeanPrecision(mean(m_μ), mean(m_γ) * diageye(samplefloattype(m_out), ndims(m_out))),
            m_out
        ),
        μ = m_μ,
        γ = m_γ
    )
end

@marginalrule MvNormalMeanScalePrecision(:out_μ_γ) (
    m_out::PointMass,
    m_μ::MultivariateNormalDistributionsFamily,
    m_γ::PointMass
) = begin
    return (
        out = m_out,
        μ = prod(
            ProdAnalytical(),
            m_μ,
            MvNormalMeanPrecision(mean(m_out), mean(m_γ) * diageye(eltype(m_out), ndims(m_out)))
        ),
        γ = m_γ
    )
end

@marginalrule MvNormalMeanScalePrecision(:out_μ_γ) (
    m_out::MultivariateNormalDistributionsFamily,
    m_μ::MultivariateNormalDistributionsFamily,
    m_γ::PointMass
) = begin
    xi_y, W_y = weightedmean_precision(m_out)
    xi_m, W_m = weightedmean_precision(m_μ)

    W_bar = mean(m_γ) * diageye(samplefloattype(m_out), ndims(m_out))

    T = promote_type(eltype(W_bar), eltype(W_y), eltype(W_m))
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

    return (out_μ = MvNormalWeightedMeanPrecision(ξ, Λ), Λ = m_Λ)
end

@marginalrule MvNormalMeanScalePrecision(:out_μ) (
    m_out::MultivariateNormalDistributionsFamily,
    m_μ::MultivariateNormalDistributionsFamily,
    q_γ::Any
) = begin
    xi_y, W_y = weightedmean_precision(m_out)
    xi_m, W_m = weightedmean_precision(m_μ)

    W_bar = mean(q_γ) * diageye(eltype(m_out), ndims(m_out))

    T = promote_type(eltype(W_bar), eltype(W_y), eltype(W_m))
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

@marginalrule MvNormalMeanScalePrecision(:out_μ) (
    m_out::PointMass,
    m_μ::MultivariateNormalDistributionsFamily,
    q_γ::Any
) =
    begin
        return (
            out = m_out,
            μ = prod(
                ProdAnalytical(),
                MvNormalMeanPrecision(mean(m_out), mean(q_γ) * diageye(samplefloattype(m_out), ndims(m_out))),
                m_μ
            )
        )
    end

@marginalrule MvNormalMeanScalePrecision(:out_μ) (
    m_out::MultivariateNormalDistributionsFamily,
    m_μ::PointMass,
    q_γ::Any
) =
    begin
        return (
            out = prod(
                ProdAnalytical(),
                MvNormalMeanPrecision(mean(m_μ), mean(q_γ) * diageye(samplefloattype(m_out), ndims(m_out))),
                m_out
            ),
            μ = m_μ
        )
    end

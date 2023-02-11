
@rule NormalMeanPrecision(:τ, Marginalisation) (q_out::Any, q_μ::Any) = begin
    θ = 2 / (var(q_out) + var(q_μ) + abs2(mean(q_out) - mean(q_μ))) 
    α = convert(typeof(θ), 1.5)
    return Gamma(α, θ)
end

@rule NormalMeanPrecision(:τ, Marginalisation) (q_out_μ::Any,) = begin
    m, V = mean_cov(q_out_μ)
    θ = 2 / (V[1, 1] - V[1, 2] - V[2, 1] + V[2, 2] + abs2(m[1] - m[2]))
    α = convert(typeof(θ), 1.5)
    return Gamma(α, θ)
end

# GP meta #
@rule NormalMeanPrecision(:τ, Marginalisation) (q_out::Any, q_μ::GaussianProcess,meta::ProcessMeta) = begin
    m_right, cov_right = mean_cov(q_μ.finitemarginal)
    mμ, vμ = m_right[meta.index],cov_right[meta.index]
    vμ = clamp(vμ[1],1e-8,huge)
    θ = 2 / (var(q_out) + vμ[1] + abs2(mean(q_out) - mμ[1]))
    α = convert(typeof(θ), 1.5)

    return Gamma(α, θ)
end

@rule NormalMeanPrecision(:τ, Marginalisation) (q_out::PointMass, m_μ::NormalMeanVariance, ) = begin 
    return @call_rule NormalMeanPrecision(:τ, Marginalisation) (q_out = q_out, q_μ = m_μ)
end

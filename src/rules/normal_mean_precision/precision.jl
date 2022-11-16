
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
    kernelf = q_μ.kernelfunction
    meanf   = q_μ.meanfunction
    test    = q_μ.testinput
    train   = q_μ.traininput
    cov_strategy = q_μ.covariance_strategy
    x_u = q_μ.inducing_input

    mμ, vμ = predictMVN(cov_strategy, kernelf,meanf,test,[train[meta.index]],m_right, cov_right,x_u) #changed here
    vμ = clamp(vμ[1],1e-8,huge)
    θ = 2 / (var(q_out) + vμ[1] + abs2(mean(q_out) - mμ[1]))
    α = convert(typeof(θ), 1.5)

    return Gamma(α, θ)
end

import StatsFuns: log2π

@node NormalMeanPrecision Stochastic [out, (μ, aliases = [mean]), (τ, aliases = [invcov, precision])]

@average_energy NormalMeanPrecision (q_out::Any, q_μ::Any, q_τ::Any) = begin
    μ_mean, μ_var     = mean_var(q_μ)
    out_mean, out_var = mean_var(q_out)
    return (log2π - mean(log, q_τ) + mean(q_τ) * (μ_var + out_var + abs2(μ_mean - out_mean))) / 2
end

@average_energy NormalMeanPrecision (q_out_μ::MultivariateNormalDistributionsFamily, q_τ::Any) = begin
    out_μ_mean, out_μ_cov = mean_cov(q_out_μ)
    return (log2π - mean(log, q_τ) + mean(q_τ) * (out_μ_cov[1, 1] + out_μ_cov[2, 2] - out_μ_cov[1, 2] - out_μ_cov[2, 1] + abs2(out_μ_mean[1] - out_μ_mean[2]))) / 2
end

@average_energy NormalMeanPrecision (q_out::Any, q_μ::GaussianProcess, q_τ::Any, meta::ProcessMeta) = begin
    m_right, cov_right = mean_cov(q_μ.finitemarginal)
    kernelf = q_μ.kernelfunction
    meanf   = q_μ.meanfunction
    test    = q_μ.testinput
    train   = q_μ.traininput
    cov_strategy = q_μ.covariance_strategy
    x_u = q_μ.inducing_input

    μ_mean, μ_var = predictMVN(cov_strategy, kernelf, meanf, test,[train[meta.index]],m_right, x_u) #changed here
    μ_var = clamp(μ_var[1],1e-8,huge)
    μ_mean = μ_mean[1]
    out_mean, out_var = mean_var(q_out)
    return (log2π - mean(log, q_τ) + mean(q_τ) * (μ_var + out_var + abs2(μ_mean - out_mean))) / 2
end
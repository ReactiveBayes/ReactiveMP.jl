import StatsFuns: log2π

@node NormalMeanVariance Stochastic [out, (μ, aliases = [mean]), (v, aliases = [var])]

@average_energy NormalMeanVariance (q_out::Any, q_μ::Any, q_v::Any) = begin
    μ_mean, μ_var     = mean_var(q_μ)
    out_mean, out_var = mean_var(q_out)
    return (log2π + mean(log, q_v) + mean(inv, q_v) * (μ_var + out_var + abs2(μ_mean - out_mean))) / 2
end

@average_energy NormalMeanVariance (q_out_μ::MultivariateNormalDistributionsFamily, q_v::Any) = begin
    out_μ_mean, out_μ_cov = mean_cov(q_out_μ)
    return (log2π + mean(log, q_v) + mean(inv, q_v) * (out_μ_cov[1, 1] + out_μ_cov[2, 2] - out_μ_cov[1, 2] - out_μ_cov[2, 1] + abs2(out_μ_mean[1] - out_μ_mean[2]))) / 2
end

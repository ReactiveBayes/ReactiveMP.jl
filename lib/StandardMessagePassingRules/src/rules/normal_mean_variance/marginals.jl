@define_marginal_update_rule(
    node = NormalMeanVariance, towards = (:out, :μ),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:μ]::UnivariateNormalDistributionsFamily, q[:v]::Any),
    body = (args) -> begin
        xi_out, W_out = weightedmean_precision(args.m[:out])
        xi_μ, W_μ = weightedmean_precision(args.m[:μ])
        W_bar = mean(inv, args.q[:v])
        MvNormalWeightedMeanPrecision([xi_out; xi_μ], [W_out + W_bar -W_bar; -W_bar W_μ + W_bar])
    end,
)

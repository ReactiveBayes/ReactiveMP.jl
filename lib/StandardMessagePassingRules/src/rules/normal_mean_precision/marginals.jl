@define_marginal_update_rule(
    node = NormalMeanPrecision, towards = (:out, :μ),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:μ]::UnivariateNormalDistributionsFamily, q[:τ]::Any),
    body = (args) -> begin
        xi_out, W_out = weightedmean_precision(args.m[:out])
        xi_μ, W_μ = weightedmean_precision(args.m[:μ])
        W_bar = mean(args.q[:τ])
        MvNormalWeightedMeanPrecision([xi_out; xi_μ], [W_out + W_bar -W_bar; -W_bar W_μ + W_bar])
    end,
)

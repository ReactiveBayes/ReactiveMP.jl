# The SoftDot rules. Belief propagation does not exist for SoftDot:
# every rule is variational, mean-field or under a structured q(y, x), which the default
# dependency scheme tells apart by what it passes, marginals, messages or the joint.

# A normal in its natural parameters, univariate for a scalar weighted mean.
weighted_mean_precision(ξ::Real, W) = NormalWeightedMeanPrecision(ξ, W)
weighted_mean_precision(ξ::AbstractVector, W) = MvNormalWeightedMeanPrecision(ξ, W)

# Towards `y`: under mean-field, N(⟨θ⟩ᵀ⟨x⟩, ⟨γ⟩⁻¹).
@define_message_update_rule(
    node = SoftDot, target = :y, args = (q[:θ]::Any, q[:x]::Any, q[:γ]::Any),
    body = (args) -> NormalMeanPrecision(mean(args.q[:θ])' * mean(args.q[:x]), mean(args.q[:γ])),
)

# Under q(y, x), from the message on `x`: the first component of AR's `y` message.
@define_message_update_rule(
    node = SoftDot, target = :y, args = (q[:θ]::Any, m[:x]::NormalDistributionsFamily, q[:γ]::Any),
    body = (args) -> y_from_x(args.m[:x], args.q[:θ], args.q[:γ]),
)

# Towards `θ`: under mean-field, weighted mean ⟨γ⟩⟨x⟩⟨y⟩ and precision ⟨γ⟩⟨xxᵀ⟩.
@define_message_update_rule(
    node = SoftDot, target = :θ, args = (q[:y]::Any, q[:x]::Any, q[:γ]::Any),
    body = (args) -> begin
        my = mean(args.q[:y])
        mx, Vx = mean_cov(args.q[:x])
        mγ = mean(args.q[:γ])
        weighted_mean_precision(mγ * mx * my, mγ * (Vx + mx * mx'))
    end,
)

# Under q(y, x), the weighted mean is ⟨γ⟩⟨xy⟩ instead.
@define_message_update_rule(
    node = SoftDot, target = :θ, args = (q[:y, :x]::MultivariateNormalDistributionsFamily, q[:γ]::Any),
    body = (args) -> begin
        my, _, mx, Vx, Vxy = split_y_x(args.q[:y, :x])
        mγ = mean(args.q[:γ])
        weighted_mean_precision((Vxy + mx * my') * mγ, mγ * (Vx + mx * mx'))
    end,
)

# Towards `x`: under mean-field, the image of the rule towards `θ`.
@define_message_update_rule(
    node = SoftDot, target = :x, args = (q[:y]::Any, q[:θ]::Any, q[:γ]::Any),
    body = (args) -> begin
        my = mean(args.q[:y])
        mθ, Vθ = mean_cov(args.q[:θ])
        mγ = mean(args.q[:γ])
        weighted_mean_precision(mγ * mθ * my, mγ * (Vθ + mθ * mθ'))
    end,
)

# Under q(y, x), from the message on `y` of variance V_y: the noise variance becomes
# V_y + 1/⟨γ⟩ along ⟨θ⟩, and ⟨γ⟩ V_θ is added to the precision.
@define_message_update_rule(
    node = SoftDot, target = :x, args = (m[:y]::UnivariateNormalDistributionsFamily, q[:θ]::Any, q[:γ]::Any),
    body = (args) -> begin
        mθ, Vθ = mean_cov(args.q[:θ])
        my, Vy = mean_cov(args.m[:y])
        mγ = mean(args.q[:γ])
        C = mθ / (Vy + inv(mγ))
        weighted_mean_precision(C * my, C * mθ' + mγ * Vθ)
    end,
)

# Towards `γ`: Γ(3/2, ⟨(y - θᵀx)²⟩ / 2), under mean-field.
@define_message_update_rule(
    node = SoftDot, target = :γ, args = (q[:y]::Any, q[:θ]::Any, q[:x]::Any),
    body = (args) -> begin
        my, Vy = mean_cov(args.q[:y])
        mθ, Vθ = mean_cov(args.q[:θ])
        mx, Vx = mean_cov(args.q[:x])
        T = promote_paramfloattype(args.q[:y], args.q[:θ], args.q[:x])
        β = (Vy + my * my') / 2
        β -= my * mθ' * mx
        β += (StandardMessagePassingRules.mul_trace(Vx, Vθ) + mθ'Vx * mθ + mx'Vθ * mx + mθ'mx * mx'mθ) / 2
        GammaShapeRate(convert(T, 3 / 2), β)
    end,
)

# Under q(y, x), the expectation takes the cross-covariance of `y` and `x`.
@define_message_update_rule(
    node = SoftDot, target = :γ, args = (q[:y, :x]::MultivariateNormalDistributionsFamily, q[:θ]::Any),
    body = (args) -> begin
        my, Vy, mx, Vx, Vxy = split_y_x(args.q[:y, :x])
        mθ, Vθ = mean_cov(args.q[:θ])
        C = StandardMessagePassingRules.rank1update(Vx, mx)
        B = StandardMessagePassingRules.rank1update(Vy, my) - 2 * dot(mθ, Vxy + mx * my) + dot(mθ, C, mθ) + StandardMessagePassingRules.mul_trace(Vθ, C)
        GammaShapeRate(convert(typeof(B), 3 // 2), B / 2)
    end,
)

# The joint q(y, x) from the messages on `y` and `x`, as AR's: their natural parameters, and the
# precision of the factor, ⟨γ⟩ [1 -⟨θ⟩ᵀ; -⟨θ⟩ ⟨θθᵀ⟩], added to theirs.
@define_marginal_update_rule(
    node = SoftDot, target = (:y, :x),
    args = (m[:y]::NormalDistributionsFamily, m[:x]::NormalDistributionsFamily, q[:θ]::Any, q[:γ]::Any),
    body = (args) -> begin
        mθ, Vθ = mean_cov(args.q[:θ])
        mγ = mean(args.q[:γ])
        ξ_y, W_y = weightedmean_precision(args.m[:y])
        ξ_x, W_x = weightedmean_precision(args.m[:x])
        W = [W_y + mγ -mγ * mθ'; -mθ * mγ W_x + mγ * Vθ + mθ * mγ * mθ']
        MvNormalWeightedMeanPrecision([ξ_y; ξ_x], W)
    end,
)

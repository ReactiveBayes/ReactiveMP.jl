# The rules of y ~ N(A x, W⁻¹), A = f(a), each in a structured `q(y, x)` and a mean-field form.
# Rows of A are linear in `a` through the Jacobians `Fs`, so E[AᵀWA] = mAᵀ⟨W⟩mA + Σᵢⱼ ⟨W⟩ᵢⱼ Fᵢ Va Fⱼᵀ.

# Towards `y`: A x pushed through the transition, with the noise ⟨W⟩⁻¹.
@define_message_update_rule(
    node = ContinuousTransition, target = :y, algorithm = CTVMP,
    args = (m[:x]::MultivariateNormalDistributionsFamily, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> begin
        mx, Vx = mean_cov(args.m[:x])
        mA = ct_matrix(algo, args.q[:a])
        MvNormalMeanCovariance(mA * mx, mA * Vx * mA' + cholinv(mean(args.q[:W])))
    end,
)

@define_message_update_rule(
    node = ContinuousTransition, target = :y, algorithm = CTVMP,
    args = (q[:x]::Any, q[:a]::Any, q[:W]::Any),
    body = (algo, args) -> MvNormalMeanPrecision(ct_matrix(algo, args.q[:a]) * mean(args.q[:x]), mean(args.q[:W])),
)

# Towards `x`: the precision mAᵀ W̃ mA plus the uncertainty of `a`, where W̃ is ⟨W⟩ combined with
# the message on `y` (Woodbury: (Wy⁻¹ + ⟨W⟩⁻¹)⁻¹ = Wy - Wy (Wy + ⟨W⟩)⁻¹ Wy), or ⟨W⟩ itself under
# mean-field.
function ct_towards_x(algo, W̃, my, q_a, q_W)
    ma, Va = mean_cov(q_a)
    mW = mean(q_W)
    Fs = jacobians(algo, ma)
    mA = ct_matrix(algo, q_a)
    Ξ = mA' * W̃ * mA
    for (i, j) in Iterators.product(eachindex(Fs), eachindex(Fs))
        Ξ += mW[j, i] * Fs[j] * Va * Fs[i]'
    end
    return MvNormalWeightedMeanPrecision(mA' * W̃ * my, Ξ)
end

@define_message_update_rule(
    node = ContinuousTransition, target = :x, algorithm = CTVMP,
    args = (m[:y]::MultivariateNormalDistributionsFamily, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> begin
        my, Wy = mean_precision(args.m[:y])
        mW = mean(args.q[:W])
        ct_towards_x(algo, Wy - Wy * cholinv(Wy + mW) * Wy, my, args.q[:a], args.q[:W])
    end,
)

@define_message_update_rule(
    node = ContinuousTransition, target = :x, algorithm = CTVMP,
    args = (q[:y]::Any, q[:a]::Any, q[:W]::Any),
    body = (algo, args) -> ct_towards_x(algo, mean(args.q[:W]), mean(args.q[:y]), args.q[:a], args.q[:W]),
)

# Towards `a`, from E[x yᵀ] and E[x xᵀ]; the Jacobians are taken at the mean of `q(a)`, which is
# why this rule reads it. The offset O of the linearised A enters as E[x (y - O x)ᵀ]; v6 left it
# out, which is exact only for an `f` linear through the origin.
function ct_towards_a(algo, Exy, Exx, q_a, q_W)
    ma = mean(q_a)
    mW = mean(q_W)
    Fs = jacobians(algo, ma)
    H = weighted_jacobians(mW, Fs)
    Exỹ = Exy - Exx * ct_offset(ct_matrix(algo, q_a), Fs, ma)'
    ξ = zeros(eltype(ma), length(ma))
    W = zeros(eltype(ma), length(ma), length(ma))
    for i in eachindex(Fs)
        ξ += Fs[i]' * Exỹ * mW[:, i]
        W += Fs[i]' * Exx * H[i]
    end
    return MvNormalWeightedMeanPrecision(ξ, W)
end

@define_message_update_rule(
    node = ContinuousTransition, target = :a, algorithm = CTVMP,
    args = (q[:y, :x]::MultivariateNormalDistributionsFamily, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> begin
        dy = size(mean(args.q[:W]), 1)
        my, _, mx, Vx, Vyx = split_y_x(args.q[:y, :x], dy)
        R = StandardMessagePassingRules
        ct_towards_a(algo, R.rank1update(Vyx', mx, my), R.rank1update(Vx, mx), args.q[:a], args.q[:W])
    end,
)

@define_message_update_rule(
    node = ContinuousTransition, target = :a, algorithm = CTVMP,
    args = (q[:y]::Any, q[:x]::Any, q[:a]::Any, q[:W]::Any),
    body = (algo, args) -> begin
        mx, Vx = mean_cov(args.q[:x])
        ct_towards_a(algo, mx * mean(args.q[:y])', StandardMessagePassingRules.rank1update(Vx, mx), args.q[:a], args.q[:W])
    end,
)

# E[(y - A x)(y - A x)ᵀ] over q(y, x) (or q(y) q(x), with no cross-covariance) and q(a), for the
# rule towards `W` and the energy: A's mean is the linearised f(m_a), its spread the Jacobians at m_a.
function ct_residual(algo, my, Vy, mx, Vx, Vyx, q_a)
    ma, Va = mean_cov(q_a)
    Fs = jacobians(algo, ma)
    mA = ct_matrix(algo, q_a)
    Exx = StandardMessagePassingRules.rank1update(Vx, mx)
    Eyx = StandardMessagePassingRules.rank1update(Vyx, my, mx)
    S = StandardMessagePassingRules.rank1update(Vy, my) - Eyx * mA' - mA * Eyx' + mA * Exx * mA'
    return S + [tr(Fs[i]' * Exx * Fs[j] * Va) for i in eachindex(Fs), j in eachindex(Fs)]
end

# Towards `W`: a Wishart with dy + 2 degrees of freedom and the inverse scale
# E[(y - A x)(y - A x)ᵀ], the energy's. v6's took the rows of A as (Fᵢ a)ᵀ there, which drops the
# offset of an affine or nonlinear `f`.
@define_message_update_rule(
    node = ContinuousTransition, target = :W, algorithm = CTVMP,
    args = (q[:y, :x]::MultivariateNormalDistributionsFamily, q[:a]::MultivariateNormalDistributionsFamily),
    body = (algo, args) -> begin
        dy = size(ct_matrix(algo, args.q[:a]), 1)
        my, Vy, mx, Vx, Vyx = split_y_x(args.q[:y, :x], dy)
        WishartFast(dy + 2, ct_residual(algo, my, Vy, mx, Vx, Vyx, args.q[:a]))
    end,
)

@define_message_update_rule(
    node = ContinuousTransition, target = :W, algorithm = CTVMP,
    args = (q[:y]::Any, q[:x]::Any, q[:a]::Any),
    body = (algo, args) -> begin
        my, Vy = mean_cov(args.q[:y])
        mx, Vx = mean_cov(args.q[:x])
        WishartFast(length(my) + 2, ct_residual(algo, my, Vy, mx, Vx, zeros(eltype(my), length(my), length(mx)), args.q[:a]))
    end,
)

# The joint q(y, x) from the messages on `y` and `x`, by its precision.
@define_marginal_update_rule(
    node = ContinuousTransition, target = (:y, :x), algorithm = CTVMP,
    args = (m[:y]::MultivariateNormalDistributionsFamily, m[:x]::MultivariateNormalDistributionsFamily, q[:a]::Any, q[:W]::Any),
    body = (algo, args) -> begin
        ma, Va = mean_cov(args.q[:a])
        mW = mean(args.q[:W])
        Fs = jacobians(algo, ma)
        mA = ct_matrix(algo, args.q[:a])
        ξy, Wy = weightedmean_precision(args.m[:y])
        ξx, Wx = weightedmean_precision(args.m[:x])
        H = weighted_jacobians(mW, Fs)
        Ξ = Wx
        for i in eachindex(Fs)
            Ξ += H[i] * Va * Fs[i]'
        end
        R = StandardMessagePassingRules
        W = [Wy + mW R.negate_inplace!(mW * mA); R.negate_inplace!(mA' * mW) Ξ + mA' * mW * mA]
        MvNormalWeightedMeanPrecision([ξy; ξx], W)
    end,
)

# ⟨-log N(y; A x, W⁻¹)⟩ = dy/2 log 2π - ⟨log det W⟩/2 + tr(⟨W⟩ E[(y - A x)(y - A x)ᵀ])/2.
# v6 wrote it with three errors, each checked against Monte Carlo in 6.5.0: ⟨log det W⟩ not
# halved, the dimension taken as half of q(y)'s (mean-field) or of q(y, x)'s (structured), and
# the identity where the uncertainty of `a` meets E[x xᵀ] = Vx + mx mxᵀ.
ct_energy(algo, my, Vy, mx, Vx, Vyx, q_a, q_W) =
    StandardMessagePassingRules.gaussian_energy(length(my), tr(mean(q_W) * ct_residual(algo, my, Vy, mx, Vx, Vyx, q_a)) - mean(logdet, q_W))

@define_average_energy(
    node = ContinuousTransition, algorithm = CTVMP,
    args = (q[:y, :x]::Any, q[:a]::Any, q[:W]::Any),
    body = (algo, args) -> begin
        dy = size(mean(args.q[:W]), 1)
        my, Vy, mx, Vx, Vyx = split_y_x(args.q[:y, :x], dy)
        ct_energy(algo, my, Vy, mx, Vx, Vyx, args.q[:a], args.q[:W])
    end,
)

@define_average_energy(
    node = ContinuousTransition, algorithm = CTVMP,
    args = (q[:y]::Any, q[:x]::Any, q[:a]::Any, q[:W]::Any),
    body = (algo, args) -> begin
        my, Vy = mean_cov(args.q[:y])
        mx, Vx = mean_cov(args.q[:x])
        ct_energy(algo, my, Vy, mx, Vx, zeros(eltype(my), length(my), length(mx)), args.q[:a], args.q[:W])
    end,
)

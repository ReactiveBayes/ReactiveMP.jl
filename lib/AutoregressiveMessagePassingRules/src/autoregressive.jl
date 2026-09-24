@doc raw"""
    AR

The autoregressive node, a Bayesian autoregressive process of order `p`:

```math
y_t \sim \mathcal{N}(\theta^\top x_t, \, \gamma^{-1})
```

where `yₜ` is the current value, `xₜ = (yₜ₋₁, …, yₜ₋ₚ)` the `p` lagged values, `θ` the
coefficients and `γ` the precision. For `p > 1`, `y` is the whole state `(yₜ, …, yₜ₋ₚ₊₁)`,
the companion matrix of `θ` shifting `x` down by one under noise on its first component only.
Also available as `Autoregressive`.

Its interfaces are `y` (alias `out`), `x`, `θ` and `γ`. Its rules run under [`ARVMP`](@ref),
which gives the variate form and the order: the node declares no algorithm of its own, and
under the default one, `DefaultAlgorithm`, no rule exists. They follow the factorisation,
`q(y, x) q(θ) q(γ)` or mean-field. [`ConjugateAR`](@ref) keeps `(θ, γ)` joint on one edge.
"""
struct AR end

const Autoregressive = AR

"""
    ARsafe()

The numerically safe mode of [`ARVMP`](@ref): the joint `q(y, x)` is formed from a precision
matrix whose noiseless components are regularised by a large finite precision.
"""
struct ARsafe end

"""
    ARunsafe()

The unregularised mode of [`ARVMP`](@ref): the joint `q(y, x)` is formed from its covariance,
through the inverse of the companion matrix, which is faster and may be numerically fragile.
"""
struct ARunsafe end

"""
    ARVMP(form, order, stype)

The algorithm of [`AR`](@ref) and [`ConjugateAR`](@ref), which the model must give: neither
node has a default. `form` is `Univariate` or `Multivariate`, the variate form of `y` and `x`;
`Univariate` is an AR(1) and forces `order` to `1`, with a warning for any other. `order` is
the order `p`, and `stype` is [`ARsafe`](@ref)`()` or [`ARunsafe`](@ref)`()`. v6 called it
`ARMeta`.

```julia
ARVMP(Multivariate, 3, ARsafe())
```
"""
struct ARVMP{F <: VariateForm, S} <: AbstractAlgorithm
    order::Int
    stype::S
end

function ARVMP(::Type{Univariate}, order, stype::S) where {S}
    order == 1 || @warn "ARVMP(Univariate, ...) has been created with order $(order); the order is forced to 1."
    return ARVMP{Univariate, S}(1, stype)
end

ARVMP(::Type{Multivariate}, order, stype::S) where {S} = ARVMP{Multivariate, S}(order, stype)

getvform(::ARVMP{F}) where {F} = F
getorder(algo::ARVMP) = algo.order
getstype(algo::ARVMP) = algo.stype

is_multivariate(algo::ARVMP) = getvform(algo) === Multivariate
is_univariate(algo::ARVMP) = getvform(algo) === Univariate

is_safe(algo::ARVMP) = getstype(algo) === ARsafe()
is_unsafe(algo::ARVMP) = getstype(algo) === ARunsafe()

@define_factor_node(node = AR, type = Stochastic, interfaces = [(:y, aliases = [:out]), :x, :θ, :γ])

# `array[ranges...]` as a view for a multivariate AR, its only entry for a univariate one.
ar_slice(::Type{Multivariate}, array, ranges...) = view(array, ranges...)
ar_slice(::Type{Univariate}, array, ranges...) = first(view(array, ranges...))

# The rules' algebra, as functions of the marginals, since ConjugateAR's rules compute the same
# from the marginals of θ and γ that its q(w) implies.

# Towards `y` from the message on `x`: the message pushed through the companion matrix of the
# mean of θ, with the variance of θ weighting its precision.
function ar_towards_y(algo::ARVMP, m_x, q_θ, q_γ)
    mθ, Vθ = mean_cov(q_θ)
    mx, Wx = mean_invcov(m_x)
    mγ = mean(q_γ)

    mA = as_companion_matrix(mθ)
    mV = ar_transition(getvform(algo), getorder(algo), mγ)

    D = Wx + mγ * Vθ
    C = mA * inv(D)

    my = C * Wx * mx
    Vy = add_transition!(C * mA', mV)

    return convert(promote_variate_type(getvform(algo), NormalMeanVariance), my, Vy)
end

# Towards `y` from q(x): its mean through the companion matrix, with the noise alone.
function ar_towards_y_meanfield(algo::ARVMP, q_x, q_θ, q_γ)
    mA = as_companion_matrix(mean(q_θ))
    mV = ar_transition(getvform(algo), getorder(algo), mean(q_γ))
    return convert(promote_variate_type(getvform(algo), NormalMeanVariance), mA * mean(q_x), mV)
end

# Towards `x` from the message on `y`.
function ar_towards_x(algo::ARVMP, m_y, q_θ, q_γ)
    mθ, Vθ = mean_cov(q_θ)
    my, Vy = mean_cov(m_y)
    mγ = mean(q_γ)

    mA = as_companion_matrix(mθ)
    mV = ar_transition(getvform(algo), getorder(algo), mγ)

    C = mA' * inv(add_transition(Vy, mV))

    W = C * mA + mγ * Vθ
    ξ = C * my

    return convert(promote_variate_type(getvform(algo), NormalWeightedMeanPrecision), ξ, W)
end

# Towards `x` from q(y).
function ar_towards_x_meanfield(algo::ARVMP, q_y, q_θ, q_γ)
    mθ, Vθ = mean_cov(q_θ)
    mγ = mean(q_γ)

    mA = as_companion_matrix(mθ)
    mV = ar_transition(getvform(algo), getorder(algo), mγ)

    C = mA' * pinv(mV)
    W = C * mA + mγ * Vθ
    ξ = C * mean(q_y)

    return convert(promote_variate_type(getvform(algo), NormalWeightedMeanPrecision), ξ, W)
end

# The joint q(y, x) from the messages on `y` and `x`, by its precision (ARsafe) or its
# covariance (ARunsafe).
ar_joint(algo::ARVMP, m_y, m_x, q_θ, q_γ) = ar_joint(getstype(algo), algo, m_y, m_x, q_θ, q_γ)

function ar_joint(::ARsafe, algo::ARVMP, m_y, m_x, q_θ, q_γ)
    mθ, Vθ = mean_cov(q_θ)
    mγ = mean(q_γ)

    mA = as_companion_matrix(mθ)
    mW = ar_precision(getvform(algo), getorder(algo), mγ)

    b_my, b_Vy = mean_cov(m_y)
    f_mx, f_Vx = mean_cov(m_x)

    inv_b_Vy = cholinv(b_Vy)
    inv_f_Vx = cholinv(f_Vx)

    D = inv_f_Vx + mγ * Vθ

    W_11 = add_precision(inv_b_Vy, mW)
    W_12 = StandardMessagePassingRules.negate_inplace!(mW * mA)   # -(mW * mA)
    W_21 = StandardMessagePassingRules.negate_inplace!(mA' * mW)  # -(mA' * mW)
    W_22 = D + mA' * mW * mA

    W = [W_11 W_12; W_21 W_22]
    ξ = [inv_b_Vy * b_my; inv_f_Vx * f_mx]

    return MvNormalWeightedMeanPrecision(ξ, W)
end

# ARunsafe: the same joint by its covariance, which needs no precision of the noise. `x` has
# precision D = Wx + ⟨γ⟩Vθ and `y = A x + noise` of covariance mV, jointly
#
#     Σ₀ = [A P Aᵀ + mV  A P; P Aᵀ  P],   μ₀ = [A μx; μx],   P = D⁻¹, μx = P Wx mx,
#
# conditioned on the message on `y` as on an observation of covariance Vy, by the Kalman gain.
# v6 wrote a block inversion through the inverse of the companion matrix whose Schur complement
# had the wrong sign (and one transpose too many), and so disagreed with ARsafe even for a
# univariate AR(1), where ARsafe regularises nothing; `test/rules/marginals_tests.jl` pins it.
function ar_joint(::ARunsafe, algo::ARVMP, m_y, m_x, q_θ, q_γ)
    mθ, Vθ = mean_cov(q_θ)
    mγ = mean(q_γ)

    mA = as_companion_matrix(mθ)
    mV = ar_transition(getvform(algo), getorder(algo), mγ)

    b_my, b_Vy = mean_cov(m_y)
    f_mx, f_Vx = mean_cov(m_x)

    inv_f_Vx = cholinv(f_Vx)
    P = cholinv(inv_f_Vx + mγ * Vθ)
    μx = P * (inv_f_Vx * f_mx)

    AP = mA * P
    Σyy = AP * mA' + mV
    G = [Σyy; AP']
    K = G * cholinv(Σyy + b_Vy)

    μ = [mA * μx; μx] + K * (b_my - mA * μx)
    Σ = [Σyy AP; AP' P] - K * G'

    return MvNormalMeanCovariance(μ, Σ)
end

# The average energy under q(y, x) q(θ) q(γ). For a multivariate AR only the first component
# of `y` is noisy: the rest is a copy of `x`, whose entropy the joint's entropy counts again and
# the correction takes out.
function ar_energy(algo::ARVMP, q_y_x, q_θ, q_γ)
    mθ, Vθ = mean_cov(q_θ)
    myx, Vyx = mean_cov(q_y_x)
    mγ = mean(q_γ)

    F = getvform(algo)
    order = getorder(algo)
    x_range = (order + 1):(2order)

    mx, Vx = ar_slice(F, myx, x_range), ar_slice(F, Vyx, x_range, x_range)
    my1, Vy1 = first(myx), first(Vyx)
    Vy1x = ar_slice(F, Vyx, 1, x_range)

    # (-⟨log γ⟩ + log 2π + ⟨γ⟩ (Vy1 + my1² - 2 mθᵀ (Vy1x + mx my1) + tr(Vθ Vx) + mxᵀ Vθ mx + mθᵀ (Vx + mx mxᵀ) mθ)) / 2
    AE = (
        -mean(log, q_γ) +
            log2π +
            mγ * (
            Vy1 + my1^2 - 2 * mθ' * (Vy1x + mx * my1) +
                StandardMessagePassingRules.mul_trace(Vθ, Vx) +
                dot(mx, Vθ, mx) +
                dot(mθ, Vx, mθ) +
                abs2(dot(mθ, mx))
        )
    ) / 2

    if is_multivariate(algo)
        # v6 built the index vector with LazyArrays.Vcat.
        noisy = [1; x_range]
        AE += entropy(q_y_x)
        AE -= entropy(MvNormalMeanCovariance(myx[noisy], Vyx[noisy, noisy]))
    end

    return AE
end

@define_message_update_rule(
    node = AR, target = :y, algorithm = ARVMP,
    args = (m[:x]::NormalDistributionsFamily, q[:θ]::Any, q[:γ]::Any),
    body = (algo, args) -> ar_towards_y(algo, args.m[:x], args.q[:θ], args.q[:γ]),
)

@define_message_update_rule(
    node = AR, target = :y, algorithm = ARVMP,
    args = (q[:x]::Any, q[:θ]::Any, q[:γ]::Any),
    body = (algo, args) -> ar_towards_y_meanfield(algo, args.q[:x], args.q[:θ], args.q[:γ]),
)

@define_message_update_rule(
    node = AR, target = :x, algorithm = ARVMP,
    args = (m[:y]::NormalDistributionsFamily, q[:θ]::Any, q[:γ]::Any),
    body = (algo, args) -> ar_towards_x(algo, args.m[:y], args.q[:θ], args.q[:γ]),
)

@define_message_update_rule(
    node = AR, target = :x, algorithm = ARVMP,
    args = (q[:y]::Any, q[:θ]::Any, q[:γ]::Any),
    body = (algo, args) -> ar_towards_x_meanfield(algo, args.q[:y], args.q[:θ], args.q[:γ]),
)

# Towards `θ` under q(y, x): the expected sufficient statistics of the regression of the first
# component of `y` on `x`, weighted by ⟨γ⟩.
@define_message_update_rule(
    node = AR, target = :θ, algorithm = ARVMP,
    args = (q[:y, :x]::MultivariateNormalDistributionsFamily, q[:γ]::Any),
    body = (algo, args) -> begin
        order = getorder(algo)
        F = getvform(algo)
        q_y_x, q_γ = args.q[:y, :x], args.q[:γ]

        myx, Vyx = mean_cov(q_y_x)
        my, mx = ar_slice(F, myx, 1:order), ar_slice(F, myx, (order + 1):(2order))
        Vx = ar_slice(F, Vyx, (order + 1):(2order), (order + 1):(2order))
        Vxy = ar_slice(F, Vyx, (order + 1):(2order), 1:order)

        mγ = mean(q_γ)

        # W = mγ * (Vx + mx * mx')
        W = StandardMessagePassingRules.mul_inplace!(mγ, StandardMessagePassingRules.rank1update(Vx, mx))

        c = ar_unit(promote_paramfloattype(q_y_x, q_γ), F, order)

        # ξ = (Vxy + mx * my') * c * mγ
        ξ = StandardMessagePassingRules.mul_inplace!(mγ, StandardMessagePassingRules.rank1update(Vxy, mx, my) * c)

        convert(promote_variate_type(F, NormalWeightedMeanPrecision), ξ, W)
    end,
)

@define_message_update_rule(
    node = AR, target = :θ, algorithm = ARVMP,
    args = (q[:y]::Any, q[:x]::Any, q[:γ]::Any),
    body = (algo, args) -> begin
        order = getorder(algo)
        F = getvform(algo)
        q_y, q_x, q_γ = args.q[:y], args.q[:x], args.q[:γ]

        mx, Vx = mean_cov(q_x)
        my, mγ = mean(q_y), mean(q_γ)

        mV = ar_transition(F, order, mγ)
        c = ar_unit(promote_paramfloattype(q_y, q_x, q_γ), F, order)

        ξ = mx * c' * pinv(mV) * my
        W = mγ * (Vx + mx * mx')

        convert(promote_variate_type(F, NormalWeightedMeanPrecision), ξ, W)
    end,
)

# Towards `γ`: a Gamma of shape 3/2 whose rate is half the expected squared residual of the
# first component.
@define_message_update_rule(
    node = AR, target = :γ, algorithm = ARVMP,
    args = (q[:y, :x]::MultivariateNormalDistributionsFamily, q[:θ]::Any),
    body = (algo, args) -> begin
        order = getorder(algo)
        F = getvform(algo)

        y_x_mean, y_x_cov = mean_cov(args.q[:y, :x])
        mθ, Vθ = mean_cov(args.q[:θ])

        mA = as_companion_matrix(mθ)
        my, Vy = ar_slice(F, y_x_mean, 1:order), ar_slice(F, y_x_cov, 1:order, 1:order)
        mx, Vx = ar_slice(F, y_x_mean, (order + 1):(2order)), ar_slice(F, y_x_cov, (order + 1):(2order), (order + 1):(2order))
        Vxy = ar_slice(F, y_x_cov, (order + 1):(2order), 1:order)

        C = StandardMessagePassingRules.rank1update(Vx, mx)
        R = StandardMessagePassingRules.rank1update(Vy, my)
        L = StandardMessagePassingRules.rank1update(Vxy, mx, my)

        B = first(R) - 2 * first(mA * L) + first(mA * C * mA') + StandardMessagePassingRules.mul_trace(Vθ, C)

        GammaShapeRate(convert(eltype(B), 3 // 2), B / 2)
    end,
)

@define_message_update_rule(
    node = AR, target = :γ, algorithm = ARVMP,
    args = (q[:y]::Any, q[:x]::Any, q[:θ]::Any),
    body = (args) -> begin
        my, Vy = mean_cov(args.q[:y])
        mx, Vx = mean_cov(args.q[:x])
        mθ, Vθ = mean_cov(args.q[:θ])

        B = first(Vy) + first(my)^2 - 2 * first(my) * mθ' * mx + mx' * Vθ * mx + mθ' * (Vx + mx * mx') * mθ

        GammaShapeRate(convert(eltype(B), 3 // 2), B / 2)
    end,
)

@define_marginal_update_rule(
    node = AR, target = (:y, :x), algorithm = ARVMP,
    args = (m[:y]::NormalDistributionsFamily, m[:x]::NormalDistributionsFamily, q[:θ]::Any, q[:γ]::Any),
    body = (algo, args) -> ar_joint(algo, args.m[:y], args.m[:x], args.q[:θ], args.q[:γ]),
)

@define_average_energy(
    node = AR, algorithm = ARVMP,
    args = (q[:y, :x]::MultivariateNormalDistributionsFamily, q[:θ]::Any, q[:γ]::Any),
    body = (algo, args) -> ar_energy(algo, args.q[:y, :x], args.q[:θ], args.q[:γ]),
)

# Under mean-field: the first component of `y` is the only noisy one, so for a multivariate AR
# the entropy of q(y) is replaced by that of its first component.
@define_average_energy(
    node = AR, algorithm = ARVMP,
    args = (q[:y]::NormalDistributionsFamily, q[:x]::NormalDistributionsFamily, q[:θ]::Any, q[:γ]::Any),
    body = (algo, args) -> begin
        q_y, q_x, q_θ, q_γ = args.q[:y], args.q[:x], args.q[:θ], args.q[:γ]
        mθ, Vθ = mean_cov(q_θ)
        my, Vy = mean_cov(q_y)
        mx, Vx = mean_cov(q_x)
        mγ = mean(q_γ)

        my1, Vy1 = first(my), first(Vy)

        AE = -0.5mean(log, q_γ) + 0.5log2π +
            0.5 * mγ * (
            Vy1 + my1^2 - 2 * mθ' * mx * my1 +
                StandardMessagePassingRules.mul_trace(Vθ, Vx) +
                dot(mx, Vθ, mx) +
                dot(mθ, Vx, mθ) +
                abs2(dot(mθ, mx))
        )

        if is_multivariate(algo)
            AE += entropy(q_y)
            AE -= entropy(NormalMeanVariance(my1, Vy1))
        end

        AE
    end,
)


# Structured conjugate marginal for the combined (θ, γ) parameter edge of `ConjugateAR`.
#
# Under the factorization q(y, x) q(θ, γ), the joint posterior over the parameters is the
# Bayesian-linear-regression Normal-Gamma posterior, with the regression statistics
# replaced by the expected sufficient statistics of q(y, x):
#
#     C = E[x xᵀ] = Vx + mx mxᵀ        (≙ XᵀX)
#     b = E[x y₁] = cov(x, y₁) + mx my₁ (≙ Xᵀy)
#     a = E[y₁²]  = Vy₁ + my₁²          (≙ yᵀy)
#
# Given the inbound prior message m_w = MvNormalGamma(μ₀, Λ₀, α₀, β₀):
#
#     Λn = Λ₀ + C
#     μn = Λn⁻¹ (Λ₀ μ₀ + b)
#     αn = α₀ + 1/2
#     βn = β₀ + ½ (a + μ₀ᵀΛ₀μ₀ − μnᵀΛnμn)
#
# This equals prod(m_w, likelihood) and reduces to the scalar NormalGamma update at
# order 1. See design/ar_mvnormalgamma_conjugate_rule.md and statproofbook.github.io/P/blr-post.
@marginalrule ConjugateAR(:w) (
    m_w::MvNormalGamma,
    q_y_x::MultivariateNormalDistributionsFamily,
    meta::ARMeta,
) = begin
    order = getorder(meta)

    myx, Vyx = mean_cov(q_y_x)

    x_idx = (order + 1):(2order)

    mx       = myx[x_idx]
    my1      = first(myx)
    Vx       = Vyx[x_idx, x_idx]
    Vy1      = Vyx[1, 1]
    cov_x_y1 = Vyx[x_idx, 1]

    # Expected sufficient statistics of the AR likelihood under q(y, x).
    C = Vx + mx * transpose(mx)
    b = cov_x_y1 + mx * my1
    a = Vy1 + my1^2

    μ0, Λ0, α0, β0 = params(m_w)

    # Bayesian-linear-regression Normal-Gamma posterior (single soft observation).
    Λn = Λ0 + C
    μn = cholinv(Λn) * (Λ0 * μ0 + b)
    αn = α0 + one(α0) / 2
    βn = β0 + (a + dot(μ0, Λ0, μ0) - dot(μn, Λn, μn)) / 2

    return MvNormalGamma(μn, Λn, αn, βn)
end

# Joint state marginal q(y, x). The factor matches AR, so delegate to the AR marginal rule with
# the effective (q_θ, q_γ) derived from the joint q(w) = MvNormalGamma.
@marginalrule ConjugateAR(:y_x) (
    m_y::NormalDistributionsFamily,
    m_x::NormalDistributionsFamily,
    q_w::MvNormalGamma,
    meta::ARMeta,
) = begin
    q_θ, q_γ = conjugatear_effective_marginals(q_w)
    return @call_marginalrule AR(:y_x) (
        m_y = m_y, m_x = m_x, q_θ = q_θ, q_γ = q_γ, meta = meta
    )
end

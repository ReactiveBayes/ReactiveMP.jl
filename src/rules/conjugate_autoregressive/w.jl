
# Variational message from the ConjugateAR factor toward the joint parameter edge w = (θ, γ).
#
# It is the AR likelihood seen as a function of (θ, γ), averaged over q(y, x): a Normal-Gamma
# factor with natural parameters (b, -C/2, 1/2, -a/2), where
#     C = E[x xᵀ],  b = E[x y₁],  a = E[y₁²].
# In mean parameters: Λ = C, μ = C⁻¹b, α = (3 - d)/2, β = (a - bᵀC⁻¹b)/2. The message can be
# improper (α ≤ 0 for order ≥ 3); the marginal q(w), formed by the equality node as the product
# of this message with the prior, is the proper MvNormalGamma posterior (natural parameters add).
@rule ConjugateAR(:w, Marginalisation) (
    q_y_x::MultivariateNormalDistributionsFamily, meta::ARMeta
) = begin
    order = getorder(meta)

    myx, Vyx = mean_cov(q_y_x)
    x_idx = (order + 1):(2order)

    mx       = myx[x_idx]
    my1      = first(myx)
    Vx       = Vyx[x_idx, x_idx]
    Vy1      = Vyx[1, 1]
    cov_x_y1 = Vyx[x_idx, 1]

    C = Vx + mx * transpose(mx)
    b = cov_x_y1 + mx * my1
    a = Vy1 + my1^2

    μ = cholinv(C) * b
    α = convert(eltype(μ), (3 - order) / 2)
    β = (a - dot(b, μ)) / 2

    return MvNormalGamma(μ, C, α, β)
end

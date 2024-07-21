# Belief propagation: does not exist for softdot

# Variational MP: Mean-field
@rule softdot(:θ, Marginalisation) (q_y::Any, q_x::Any, q_γ::Any) = begin
    my = mean(q_y)
    mx, Vx = mean_cov(q_x)
    mγ = mean(q_γ)
    Dθ = mγ * (Vx + mx * mx')
    zθ = mγ * mx * my
    return convert(promote_variate_type(variate_form(typeof(q_x)), NormalWeightedMeanPrecision), zθ, Dθ)
end

# Variational MP: Structured
@rule softdot(:θ, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_γ::Any) = begin
    # q_y is always Univariate
    order = length(q_y_x) - 1
    F     = order == 1 ? Univariate : Multivariate

    myx, Vyx = mean_cov(q_y_x)
    my, Vy   = first(myx), first(Vyx)
    mx, Vx   = ar_slice(F, myx, 2:(order + 1)), ar_slice(F, Vyx, 2:(order + 1), 2:(order + 1))
    Vyx      = ar_slice(F, Vyx, 2:(order + 1))

    mγ = mean(q_γ)

    W = mγ * (Vx + mx * mx')

    ξ = (Vyx + mx * my') * mγ

    return convert(promote_variate_type(F, NormalWeightedMeanPrecision), ξ, W)
end

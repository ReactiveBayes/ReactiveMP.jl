# Belief propagation: does not exist for softdot

# Variational MP
@rule softdot(:x, Marginalisation) (q_y::NormalDistributionsFamily, q_θ::MultivariateNormalDistributionsFamily, q_γ::GammaDistributionsFamily) = begin
    my = mean(q_y)
    mθ, Vθ = mean_cov(q_θ)
    mγ = mean(q_γ)
    Dx = mγ * (Vθ + mθ*transpose(mθ))
    zx = mγ * mθ * transpose(my)
    inv_Dx = inv(Dx)
    return MvNormalMeanPrecision(inv_Dx * zx, inv_Dx)
end

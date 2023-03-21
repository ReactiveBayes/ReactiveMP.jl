# Belief propagation: does not exist for softdot

# Variational MP
@rule softdot(:θ, Marginalisation) (q_y::NormalDistributionsFamily, q_x::MultivariateNormalDistributionsFamily, q_γ::GammaDistributionsFamily) = begin
    my = mean(q_y)
    mx, Vx = mean_cov(q_x)
    mγ = mean(q_γ)
    Dθ = mγ * (Vx + mx*transpose(mx))
    zθ = mγ * mx * my
    inv_Dθ = inv(Dθ)
    return MvNormalMeanPrecision(inv_Dθ * zθ, inv_Dθ)
end

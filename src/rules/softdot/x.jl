# Belief propagation: does not exist for softdot

# Variational MP
@rule softdot(:x, Marginalisation) (q_y::Any, q_θ::Any, q_γ::Any) = begin
    my = mean(q_y)
    mθ, Vθ = mean_cov(q_θ)
    mγ = mean(q_γ)
    Dx = mγ*(Vθ + mθ*mθ')
    zx = mγ*mθ*my'
    return MvNormalWeightedMeanPrecision(zx, Dx)
end

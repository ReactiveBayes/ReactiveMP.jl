import LinearAlgebra: transpose
# Belief propagation: does not exist for softdot

# Variational MP
@rule softdot(:out, Marginalisation) (q_θ::MultivariateNormalDistributionsFamily, q_x::MultivariateNormalDistributionsFamily, q_γ::GammaDistributionsFamily) = begin
    mθ = mean(q_θ)
    mx = mean(q_x)
    mγ = mean(q_γ)
    return NormalMeanPrecision(transpose(mθ) * mx, inv(mγ))
end

# Belief propagation: does not exist for softdot

# Variational MP
@rule softdot(:θ, Marginalisation) (q_y::Any, q_x::Any, q_γ::Any) = begin
    my = mean(q_y)
    mx, Vx = mean_cov(q_x)
    mγ = mean(q_γ)
    Dθ = mγ * (Vx + mx * mx')
    zθ = mγ * mx * my
    return convert(promote_variate_type(variate_form(q_x), NormalWeightedMeanPrecision), zθ, Dθ)
end

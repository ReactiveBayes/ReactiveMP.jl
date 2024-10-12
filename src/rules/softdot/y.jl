# Belief propagation: does not exist for softdot

# Variational MP: Mean-field
@rule softdot(:y, Marginalisation) (q_θ::Any, q_x::Any, q_γ::Any) = NormalMeanPrecision(mean(q_θ)'mean(q_x), mean(q_γ))

@rule softdot(:y, Marginalisation) (q_θ::Any, m_x::Any, q_γ::Any) = NormalMeanVariance(
    first.(mean_cov((@call_rule AR(:y, Marginalisation) (m_x = m_x, q_θ = q_θ, q_γ = q_γ, meta = ARMeta(variate_form(typeof(m_x)), length(q_θ), ARsafe())))))...
)

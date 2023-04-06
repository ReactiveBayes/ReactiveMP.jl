# Belief propagation: does not exist for softdot

# Variational MP: Mean-field
@rule softdot(:y, Marginalisation) (q_θ::Any, q_x::Any, q_γ::Any) = NormalMeanPrecision(mean(q_θ)'mean(q_x), mean(q_γ))

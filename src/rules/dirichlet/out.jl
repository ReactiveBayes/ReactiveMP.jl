export rule

@rule Dirichlet(:out, Marginalisation) (m_a::PointMass{ <: AbstractVector }, ) = Dirichlet(mean(m_a))

@rule Dirichlet(:out, Marginalisation) (q_a::PointMass{ <: AbstractVector }, ) = Dirichlet(mean(q_a))
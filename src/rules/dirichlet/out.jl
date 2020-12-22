export rule

@rule Dirichlet(:out, Marginalisation) (m_a::Dirac{ <: AbstractVector }, ) = Dirichlet(mean(m_a))

@rule Dirichlet(:out, Marginalisation) (q_a::Dirac{ <: AbstractVector }, ) = Dirichlet(mean(q_a))
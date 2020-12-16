export rule

@rule Dirichlet(:out, Marginalisation) (m_a::Dirac{ <: AbstractVector }, ) = Dirichlet(mean(m_a))
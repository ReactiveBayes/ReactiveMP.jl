export rule

@rule BIFM_helper(:output, Marginalisation) (q_input::Any, ) = Marginal(q_input)
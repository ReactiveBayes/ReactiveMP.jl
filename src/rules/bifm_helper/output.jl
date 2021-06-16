export rule

@rule BIFM_helper(:output, Marginalisation) (q_input::Any, ) = MarginalDistribution(q_input)
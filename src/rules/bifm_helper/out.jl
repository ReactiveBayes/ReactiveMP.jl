export rule

@rule BIFM_helper(:out, Marginalisation) (q_in::Any, ) = MarginalDistribution(q_in)
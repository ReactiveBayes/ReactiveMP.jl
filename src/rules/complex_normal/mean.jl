export rule

@rule ComplexNormal(:μ, Marginalisation) (q_out::PointMass, q_Γ::PointMass, q_C::PointMass) = ComplexNormal(mean(q_out), mean(q_Γ), mean(q_C))

@rule ComplexNormal(:μ, Marginalisation) (q_out::Any, q_Γ::Any, q_C::Any) = ComplexNormal(mean(q_out), mean(q_Γ), mean(q_C))

@rule ComplexNormal(:μ, Marginalisation) (q_out::ComplexNormal, q_Γ::Any, q_C::Any) = ComplexNormal(mean(q_out), cov(q_out) + mean(q_Γ), mean(q_C))
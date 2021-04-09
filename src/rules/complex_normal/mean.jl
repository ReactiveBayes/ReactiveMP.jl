export rule

@rule ComplexNormal(:μ, Marginalisation) (q_out::PointMass, q_Γ::PointMass, q_C::PointMass) = ComplexNormal(mean(q_μ), mean(q_Γ), mean(q_C))

@rule ComplexNormal(:μ, Marginalisation) (q_out::Any, q_Γ::Any, q_C::Any) = ComplexNormal(mean(q_μ), mean(q_Γ), mean(q_C))

@rule ComplexNormal(:μ, Marginalisation) (q_out::ComplexNormal, q_Γ::Any, q_C::Any) = ComplexNormal(mean(q_μ), cov(q_μ) + mean(q_Γ), mean(q_C))
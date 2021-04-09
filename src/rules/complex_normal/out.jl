export rule

@rule ComplexNormal(:out, Marginalisation) (q_μ::PointMass, q_Γ::PointMass, q_C::PointMass) = ComplexNormal(mean(q_μ), mean(q_Γ), mean(q_C))

@rule ComplexNormal(:out, Marginalisation) (q_μ::Any, q_Γ::Any, q_C::Any) = ComplexNormal(mean(q_μ), mean(q_Γ), mean(q_C))

@rule ComplexNormal(:out, Marginalisation) (q_μ::ComplexNormal, q_Γ::Any, q_C::Any) = ComplexNormal(mean(q_μ), cov(q_μ) + mean(q_Γ), mean(q_C))
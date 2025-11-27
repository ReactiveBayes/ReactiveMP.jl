@rule MvNormalWishart(:out, Marginalisation) (q_μ::PointMass, q_W::PointMass, q_λ::PointMass, q_ν::PointMass) = MvNormalWishart(mean(q_μ), mean(q_W), mean(q_λ), mean(q_ν))

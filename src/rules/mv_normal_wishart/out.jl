@rule MvNormalWishart(:out, Marginalisation) (q_μ::Any, q_W::Any, q_λ::Any, q_ν::Any) = MvNormalWishart(mean(q_μ), mean(q_W), mean(q_λ), mean(q_ν))

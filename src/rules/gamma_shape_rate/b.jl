export rule

@rule GammaShapeRate(:β, Marginalisation) (q_out::Any, q_α::Any) = GammaShapeRate(1 + mean(q_α), mean(q_out))

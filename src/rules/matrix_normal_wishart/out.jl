
@rule MatrixNormalWishart(:out, Marginalisation) (q_M::PointMass, q_U::PointMass, q_V::PointMass, q_ν::PointMass) = MatrixNormalWishart(
    mean(q_M), mean(q_U), mean(q_V), mean(q_ν)
)

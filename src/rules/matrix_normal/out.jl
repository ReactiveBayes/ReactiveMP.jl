@rule MatrixNormal(:out, Marginalisation) (q_M::PointMass, q_U::PointMass, q_V::PointMass) = begin
    MvNormalMeanCovariance(vec(mean(q_M)), kron(mean(q_U), mean(q_V)))
end

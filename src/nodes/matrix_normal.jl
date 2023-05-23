@node MatrixNormal Stochastic [out, M, U, V]

# default method for mean-field assumption
@average_energy MatrixNormal (q_out::Any, q_M::Any, q_U::Any, q_V::Any) = begin
    q_Σ = PointMass(kron(mean(q_U), mean(q_V)))
    -score(AverageEnergy(), MvNormalMeanCovariance, Val{(:out, :μ, :Σ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_M, q_Σ)), nothing)
end
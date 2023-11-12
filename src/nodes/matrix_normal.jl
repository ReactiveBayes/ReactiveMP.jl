@node MatrixNormal Stochastic [out, M, U, V]

# we use equivalence of ``
@average_energy MatrixNormal (q_out::Any, q_M::Any, q_U::Any, q_V::Any) = begin
    q_Σ = PointMass(kron(mean(q_U), mean(q_V)))
    q_m = PointMass(vec(mean(q_M)))
    -score(AverageEnergy(), MvNormalMeanCovariance, Val{(:out, :μ, :Σ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m, q_Σ)), nothing)
end

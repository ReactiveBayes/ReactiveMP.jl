@rule(
    formtype    => NormalMixture,
    on          => :switch,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_m::NTuple{2, NormalMeanVariance}, q_p::NTuple{2, Gamma}),
    meta        => Nothing,
    begin
        U1 = -score(AverageEnergy(), NormalMeanPrecision, Val{ (:out, :μ, :τ) }, map(as_marginal, (q_out, q_m[1], q_p[1])), nothing)
        U2 = -score(AverageEnergy(), NormalMeanPrecision, Val{ (:out, :μ, :τ) }, map(as_marginal, (q_out, q_m[2], q_p[2])), nothing)
        return Bernoulli(softmax((U1, U2))[1])
    end
)
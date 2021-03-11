
@rule GammaMixture{N}(:switch, Marginalisation) (q_out::Any, q_a::NTuple{N, Any }, q_b::NTuple{N, GammaDistributionsFamily }) where { N } = begin

    U = map(zip(q_a, q_b)) do (a, b)
        return -score(AverageEnergy(), GammaShapeRate, Val{ (:out, :α, :β) }, map(as_marginal, (q_out, a, b)), nothing)
    end

    ρ = clamp.(softmax(U), tiny, 1.0 - tiny)
    ρ = ρ ./ sum(ρ)

    return Categorical(ρ)
end

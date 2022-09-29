
@rule GammaMixture{N}(:switch, Marginalisation) (
    q_out::Any,
    q_a::NTuple{N, Any},
    q_b::NTuple{N, GammaDistributionsFamily}
) where {N} = begin
    U = map(zip(q_a, q_b)) do (a, b)
        -score(
            AverageEnergy(),
            GammaShapeRate,
            Val{(:out, :α, :β)},
            map((q) -> Marginal(q, false, false), (q_out, a, b)),
            nothing
        )
    end

    ρ = clamp.(softmax(U), tiny, one(eltype(U)) - tiny)
    ρ = ρ ./ sum(ρ)

    return Categorical(ρ)
end

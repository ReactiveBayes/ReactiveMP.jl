export rule

# NOTE: We removed this rule in favor of Cateforical distribution, Gaussian Mixture node always returns Categorical for switch distribution
# In case of Bernoulli prior later on multiplication of Categorical and Bernoulli results in Bernoulli posterior marginal
# @rule NormalMixture{2}(:switch, Marginalisation) (q_out::Any, q_m::NTuple{2, NormalMeanVariance}, q_p::NTuple{2, Gamma}) = begin
#     U1 = -score(AverageEnergy(), NormalMeanPrecision, Val{ (:out, :μ, :τ) }, map(as_marginal, (q_out, q_m[1], q_p[1])), nothing)
#     U2 = -score(AverageEnergy(), NormalMeanPrecision, Val{ (:out, :μ, :τ) }, map(as_marginal, (q_out, q_m[2], q_p[2])), nothing)
#     return Bernoulli(clamp(softmax((U1, U2))[1], tiny, 1.0 - tiny))
# end

@rule NormalMixture{N}(:switch, Marginalisation) (q_out::Any, q_m::NTuple{N, UnivariateNormalDistributionsFamily}, q_p::NTuple{N, GammaDistributionsFamily}) where { N } = begin
    U = map(zip(q_m, q_p)) do (m, p)
        return -score(AverageEnergy(), NormalMeanPrecision, Val{ (:out, :μ, :τ) }, map((q) -> Marginal(q, false , false), (q_out, m, p)), nothing)
    end
    return Categorical(clamp.(softmax(U), tiny, one(eltype(U)) - tiny))
end

@rule NormalMixture{N}(:switch, Marginalisation) (q_out::Any, q_m::NTuple{N, MultivariateNormalDistributionsFamily}, q_p::NTuple{N, Wishart}) where { N } = begin
    U = map(zip(q_m, q_p)) do (m, p)
        return -score(AverageEnergy(), MvNormalMeanPrecision, Val{ (:out, :μ, :Λ) }, map((q) -> Marginal(q, false, false), (q_out, m, p)), nothing)
    end
    return Categorical(clamp.(softmax(U), tiny, one(eltype(U)) - tiny))
end

@rule NormalMixture{N}(:switch, Marginalisation) (q_out::Any, q_m::NTuple{N, PointMass{T} where T <: Real}, q_p::NTuple{N, PointMass{T} where T <: Real}) where { N } = begin
    U = map(zip(q_m, q_p)) do (m, p)
        return -score(AverageEnergy(), NormalMeanPrecision, Val{ (:out, :μ, :τ) }, map((q) -> Marginal(q, false , false), (q_out, m, p)), nothing)
    end
    return Categorical(clamp.(softmax(U), tiny, one(eltype(U)) - tiny))
end

@rule NormalMixture{N}(:switch, Marginalisation) (q_out::Any, q_m::NTuple{N, PointMass{T} where T <: AbstractVector}, q_p::NTuple{N, PointMass{T} where T <: AbstractMatrix}) where { N } = begin
    U = map(zip(q_m, q_p)) do (m, p)
        return -score(AverageEnergy(), MvNormalMeanPrecision, Val{ (:out, :μ, :Λ) }, map((q) -> Marginal(q, false , false), (q_out, m, p)), nothing)
    end
    return Categorical(clamp.(softmax(U), tiny, one(eltype(U)) - tiny))
end
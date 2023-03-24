export rule

# NOTE: We removed this rule in favor of Cateforical distribution, Gaussian Mixture node always returns Categorical for switch distribution
# In case of Bernoulli prior later on multiplication of Categorical and Bernoulli results in Bernoulli posterior marginal
# @rule NormalMixture{2}(:switch, Marginalisation) (q_out::Any, q_m::NTuple{2, NormalMeanVariance}, q_p::NTuple{2, Gamma}) = begin
#     U1 = -score(AverageEnergy(), NormalMeanPrecision, Val{ (:out, :μ, :τ) }, map(as_marginal, (q_out, q_m[1], q_p[1])), nothing)
#     U2 = -score(AverageEnergy(), NormalMeanPrecision, Val{ (:out, :μ, :τ) }, map(as_marginal, (q_out, q_m[2], q_p[2])), nothing)
#     return Bernoulli(clamp(softmax((U1, U2))[1], tiny, 1.0 - tiny))
# end

@rule NormalMixture{N}(:switch, Marginalisation) (q_out::Any, q_m::ManyOf{N, Any}, q_p::ManyOf{N, Any}) where {N} = begin
    U = map(zip(q_m, q_p)) do (m, p)
        rule_nm_switch_k(variate_form(m), q_out, m, p)
    end
    return Categorical(clamp!(softmax!(U), tiny, one(eltype(U)) - tiny))
end

function rule_nm_switch_k(::Type{Univariate}, q_out, m, p)
    return -score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, m, p)), nothing)
end

function rule_nm_switch_k(::Type{Multivariate}, q_out, m, p)
    return -score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out, :μ, :Λ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, m, p)), nothing)
end

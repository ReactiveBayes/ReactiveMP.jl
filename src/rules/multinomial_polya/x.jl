
@rule MultinomialPolya(:x, Marginalisation) (q_N::Union{PointMass, Poisson, Binomial, Categorical}, q_ψ::Any, meta::Union{MultinomialPolyaMeta, Nothing}) = begin
    N = mode(q_N)
    m_ψ = mean(q_ψ)
    p = logistic_stick_breaking(m_ψ)
    return Multinomial(N, p)
end

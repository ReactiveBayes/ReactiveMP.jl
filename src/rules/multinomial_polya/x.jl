
@rule MultinomialPolya(:x, Marginalisation) (q_N::PointMass, q_ψ::GaussianDistributionsFamily, meta::Union{MultinomialPolyaMeta, Nothing}) = begin
    N = mean(q_N)
    m_ψ = mean(q_ψ)
    p = logistic_stic_breaking(m_ψ)
    return Multinomial(N, p)
end
